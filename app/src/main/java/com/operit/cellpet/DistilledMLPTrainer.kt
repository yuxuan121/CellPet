package com.operit.cellpet

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.*

/**
 * 蒸馏式 MLP 训练器 (6→24→12→6)。
 *
 * 与旧版 MLPTrainer 的核心区别：
 *   "总结" → "蒸馏": 不是记忆输入→输出，而是提取可复用的决策模块。
 *
 * 蒸馏流水线:
 *   1. 数据蒸馏: 质量分层 (gold/silver/bronze/counter) + 去噪 + 聚类
 *   2. 课程学习: 金标→银标→铜标→全量，渐进式训练
 *   3. 规则提取: 每卦决策边界 + hex_weights 对比
 *   4. 影子部署: 与 hex_weights 并行预测，对比分歧
 *
 * 输入: 6维 (atp, glucose, damage, cortisol, dopamine, children)
 * 输出: 6维 (6个变爻行为 logits)
 */
class DistilledMLPTrainer {

    // ==================== 网络架构 ====================
    companion object {
        const val INPUT_DIM = 6
        const val HIDDEN1 = 24
        const val HIDDEN2 = 12
        const val OUTPUT_DIM = 6

        // 归一化边界 (min-max)
        val MIN_VALS = floatArrayOf(0f, 0f, 0f, 0f, 0f, 0f)       // atp,glucose,damage,cortisol,dopamine,children
        val MAX_VALS = floatArrayOf(10f, 20f, 1f, 5f, 3f, 10f)

        // 行为名称
        val BEHAVIORS = arrayOf("divide", "repair", "store", "divide", "wait", "apoptose")
    }

    // 权重 (按旧版兼容格式: w[out][in])
    var w1: Array<FloatArray> = emptyArray(); var b1 = FloatArray(0)
    var w2: Array<FloatArray> = emptyArray(); var b2 = FloatArray(0)
    var w3: Array<FloatArray> = emptyArray(); var b3 = FloatArray(0)

    init {
        val rng = kotlin.random.Random(42)
        // He init
        val scale1 = sqrt(2f / INPUT_DIM)
        val scale2 = sqrt(2f / HIDDEN1)
        val scale3 = sqrt(2f / HIDDEN2)
        w1 = Array(HIDDEN1) { FloatArray(INPUT_DIM) { rng.nextFloat() * scale1 * 2 - scale1 } }
        b1 = FloatArray(HIDDEN1)
        w2 = Array(HIDDEN2) { FloatArray(HIDDEN1) { rng.nextFloat() * scale2 * 2 - scale2 } }
        b2 = FloatArray(HIDDEN2)
        w3 = Array(OUTPUT_DIM) { FloatArray(HIDDEN2) { rng.nextFloat() * scale3 * 2 - scale3 } }
        b3 = FloatArray(OUTPUT_DIM)
    }

    // ==================== 归一化 ====================
    fun normalize(raw: FloatArray): FloatArray {
        return FloatArray(INPUT_DIM) { i ->
            ((raw[i] - MIN_VALS[i]) / max(MAX_VALS[i] - MIN_VALS[i], 1e-8f)).coerceIn(0f, 1f)
        }
    }

    // ==================== 推理 ====================
    fun forward(xNorm: FloatArray): FloatArray {
        val h1 = FloatArray(HIDDEN1) { i ->
            var s = b1[i]; for (j in 0 until INPUT_DIM) s += w1[i][j] * xNorm[j]
            max(0f, s)
        }
        val h2 = FloatArray(HIDDEN2) { i ->
            var s = b2[i]; for (j in 0 until HIDDEN1) s += w2[i][j] * h1[j]
            max(0f, s)
        }
        val logits = FloatArray(OUTPUT_DIM) { i ->
            var s = b3[i]; for (j in 0 until HIDDEN2) s += w3[i][j] * h2[j]; s
        }
        val mx = logits.maxOrNull()!!
        val exp = FloatArray(OUTPUT_DIM) { exp(logits[it] - mx) }
        val sumExp = exp.sum()
        return FloatArray(OUTPUT_DIM) { exp[it] / sumExp }
    }

    fun predict(features: FloatArray): Int {
        val p = forward(normalize(features))
        return p.indices.maxByOrNull { p[it] }!!
    }

    fun predictWithConfidence(features: FloatArray): Pair<Int, Float> {
        val p = forward(normalize(features))
        val idx = p.indices.maxByOrNull { p[it] }!!
        return Pair(idx, p[idx])
    }

    // ==================== Mini-batch 训练 ====================
    fun trainBatch(
        xsNorm: List<FloatArray>, targets: IntArray, sampleWeights: FloatArray?,
        lr: Float = 0.01f, l2Lambda: Float = 1e-4f
    ): Pair<Float, Float> {
        val n = xsNorm.size
        if (n == 0) return Pair(0f, 0f)

        // Forward all samples
        val a1s = Array(n) { FloatArray(HIDDEN1) }
        val a2s = Array(n) { FloatArray(HIDDEN2) }
        val probs = Array(n) { FloatArray(OUTPUT_DIM) }

        for (k in 0 until n) {
            val x = xsNorm[k]
            for (i in 0 until HIDDEN1) { var s = b1[i]; for (j in 0 until INPUT_DIM) s += w1[i][j] * x[j]; a1s[k][i] = max(0f, s) }
            for (i in 0 until HIDDEN2) { var s = b2[i]; for (j in 0 until HIDDEN1) s += w2[i][j] * a1s[k][j]; a2s[k][i] = max(0f, s) }
            var mx = Float.NEGATIVE_INFINITY
            val logits = FloatArray(OUTPUT_DIM) { i -> var s = b3[i]; for (j in 0 until HIDDEN2) s += w3[i][j] * a2s[k][j]; s }
            mx = logits.maxOrNull()!!
            var sumExp = 0f
            for (i in 0 until OUTPUT_DIM) { probs[k][i] = exp(logits[i] - mx); sumExp += probs[k][i] }
            for (i in 0 until OUTPUT_DIM) probs[k][i] /= sumExp
        }

        // Gradients
        val dW1 = Array(HIDDEN1) { FloatArray(INPUT_DIM) }; val db1 = FloatArray(HIDDEN1)
        val dW2 = Array(HIDDEN2) { FloatArray(HIDDEN1) }; val db2 = FloatArray(HIDDEN2)
        val dW3 = Array(OUTPUT_DIM) { FloatArray(HIDDEN2) }; val db3 = FloatArray(OUTPUT_DIM)
        var totalLoss = 0f; var correct = 0
        var totalWeight = 0f

        for (k in 0 until n) {
            val x = xsNorm[k]; val a1 = a1s[k]; val a2 = a2s[k]
            val prob = probs[k]; val t = targets[k]
            val sw = sampleWeights?.get(k) ?: 1f
            totalWeight += sw

            val pLabel = prob[t].coerceIn(1e-8f, 1f)
            totalLoss -= ln(pLabel) * sw

            val maxIdx = prob.indices.maxByOrNull { prob[it] }!!
            if (maxIdx == t) correct += (if (sw > 0f) 1 else 0)

            // dz3 = sw * (probs - onehot) / totalWeight
            val invN = sw / max(totalWeight, 1f)
            val dz3 = FloatArray(OUTPUT_DIM) { prob[it] * invN }
            dz3[t] -= invN

            for (i in 0 until OUTPUT_DIM) {
                for (j in 0 until HIDDEN2) dW3[i][j] += dz3[i] * a2[j]
                db3[i] += dz3[i]
            }

            val dz2 = FloatArray(HIDDEN2)
            for (j in 0 until HIDDEN2) {
                var da = 0f; for (i in 0 until OUTPUT_DIM) da += dz3[i] * w3[i][j]
                dz2[j] = if (a2[j] > 0f) da else 0f
            }
            for (i in 0 until HIDDEN2) {
                for (j in 0 until HIDDEN1) dW2[i][j] += dz2[i] * a1[j]
                db2[i] += dz2[i]
            }

            val dz1 = FloatArray(HIDDEN1)
            for (j in 0 until HIDDEN1) {
                var da = 0f; for (i in 0 until HIDDEN2) da += dz2[i] * w2[i][j]
                dz1[j] = if (a1[j] > 0f) da else 0f
            }
            for (i in 0 until HIDDEN1) {
                for (j in 0 until INPUT_DIM) dW1[i][j] += dz1[i] * x[j]
                db1[i] += dz1[i]
            }
        }

        if (totalWeight < 1e-8f) return Pair(0f, 0f)
        totalLoss /= totalWeight
        val acc = correct.toFloat() / n

        // Apply with L2
        for (i in 0 until OUTPUT_DIM) {
            for (j in 0 until HIDDEN2) w3[i][j] -= lr * (dW3[i][j] + l2Lambda * w3[i][j])
            b3[i] -= lr * db3[i]
        }
        for (i in 0 until HIDDEN2) {
            for (j in 0 until HIDDEN1) w2[i][j] -= lr * (dW2[i][j] + l2Lambda * w2[i][j])
            b2[i] -= lr * db2[i]
        }
        for (i in 0 until HIDDEN1) {
            for (j in 0 until INPUT_DIM) w1[i][j] -= lr * (dW1[i][j] + l2Lambda * w1[i][j])
            b1[i] -= lr * db1[i]
        }

        return Pair(totalLoss, acc)
    }

    // ==================== 蒸馏式训练 ====================

    data class DistillStatus(
        val phase: Int,           // 0..3
        val phaseName: String,
        val epoch: Int,
        val totalEpochs: Int,
        val loss: Float,
        val accuracy: Float,
        val sampleCount: Int
    )

    data class TrainResult(
        val finalLoss: Float,
        val finalAccuracy: Float,
        val validationAccuracy: Float,
        val rulesExtracted: Int
    )

    data class DecisionRule(
        val hexagram: Int,
        val guaName: String,
        val mlpLine: Int,
        val mlpBehavior: String,
        val mlpConfidence: Float,
        val hexWeightsLine: Int,
        val hexWeightsBehavior: String,
        val agree: Boolean
    )

    data class DistilledSample(
        val features: FloatArray,    // 6维原始值
        val behaviorLabel: Int,      // 0-5
        val qualityTier: Int,        // 3=gold, 2=silver, 1=bronze, 0=counter
        val sampleWeight: Float,
        val hexagram: Int,
        val feedbackType: String
    )

    /**
     * 从 cell_experience.jsonl 加载并蒸馏数据
     */
    fun loadAndDistill(expFile: File): List<DistilledSample> {
        if (!expFile.exists()) return emptyList()

        val rawRecords = mutableListOf<Map<String, String>>()
        try {
            expFile.readLines().forEach { line ->
                if (line.isBlank()) return@forEach
                val map = parseSimpleJson(line)
                if (map.isNotEmpty()) rawRecords.add(map)
            }
        } catch (_: Exception) {}

        if (rawRecords.isEmpty()) return emptyList()

        // 质量分层
        val samples = mutableListOf<DistilledSample>()
        for (r in rawRecords) {
            val fb = r["feedback"] ?: "unknown"
            val (tier, weight) = when {
                fb == "user_feed" -> 3 to 1.0f
                fb == "user_soothe" -> 2 to 0.7f
                fb == "natural_pos" -> 1 to 0.3f
                fb == "natural_neg" -> 0 to 0.1f  // counter-example, low weight
                else -> 1 to 0.2f
            }

            val feat = floatArrayOf(
                (r["atp"]?.toFloatOrNull() ?: 5f),
                (r["glucose"]?.toFloatOrNull() ?: 10f),
                (r["damage"]?.toFloatOrNull() ?: 0.1f),
                (r["cortisol"]?.toFloatOrNull() ?: 0.5f),
                (r["dopamine"]?.toFloatOrNull() ?: 1.5f),
                ((r["children"]?.toIntOrNull() ?: 0).toFloat())
            )
            val line = r["line"]?.toIntOrNull() ?: continue
            val hex = r["hex"]?.toIntOrNull() ?: 0

            samples.add(DistilledSample(feat, line, tier, weight, hex, fb))
        }

        // 去噪: 移除孤例（该卦象下仅出现一次的行为模式）
        val hexLineCount = mutableMapOf<Pair<Int, Int>, Int>()
        for (s in samples) {
            val key = Pair(s.hexagram, s.behaviorLabel)
            hexLineCount[key] = (hexLineCount[key] ?: 0) + 1
        }
        val filtered = samples.filter { s ->
            val key = Pair(s.hexagram, s.behaviorLabel)
            (hexLineCount[key] ?: 0) >= 2 || s.qualityTier >= 2  // gold/silver always kept
        }

        return filtered
    }

    /**
     * 课程学习: 金标→银标→铜标→全量
     */
    fun trainDistilled(
        samples: List<DistilledSample>,
        onProgress: ((DistillStatus) -> Unit)? = null
    ): TrainResult {
        if (samples.isEmpty()) return TrainResult(0f, 0f, 0f, 0)

        // 80/20 分割
        val shuffled = samples.shuffled(kotlin.random.Random(42))
        val splitIdx = (shuffled.size * 0.8).toInt()
        val trainSet = shuffled.subList(0, splitIdx)
        val valSet = shuffled.subList(splitIdx, shuffled.size)

        // 课程阶段定义
        data class Phase(val id: Int, val name: String, val filter: (DistilledSample) -> Boolean,
                         val epochs: Int, val lr: Float)

        val phases = listOf(
            Phase(0, "金标核心", { it.qualityTier >= 3 }, 20, 0.02f),
            Phase(1, "银标补充", { it.qualityTier >= 2 }, 15, 0.01f),
            Phase(2, "铜标泛化", { it.qualityTier >= 1 }, 10, 0.005f),
            Phase(3, "全量精调", { true },                  5,  0.002f)
        )

        var bestValAcc = 0f

        for (phase in phases) {
            val phaseSamples = trainSet.filter(phase.filter)
            if (phaseSamples.size < 5) continue  // 样本太少跳过

            val rng = kotlin.random.Random(System.currentTimeMillis())
            val batchSize = min(16, max(4, phaseSamples.size / 5))

            for (ep in 0 until phase.epochs) {
                val batch = phaseSamples.shuffled(rng)
                var epLoss = 0f; var epAcc = 0f; var cnt = 0

                for (i in batch.indices step batchSize) {
                    val end = min(i + batchSize, batch.size)
                    val slice = batch.subList(i, end)
                    val xsNorm = slice.map { normalize(it.features) }
                    val ys = IntArray(slice.size) { j -> slice[j].behaviorLabel }
                    val ws = FloatArray(slice.size) { j -> slice[j].sampleWeight }
                    val (loss, acc) = trainBatch(xsNorm, ys, ws, phase.lr)
                    epLoss += loss * slice.size; epAcc += acc * slice.size; cnt += slice.size
                }

                if (cnt > 0) { epLoss /= cnt; epAcc /= cnt }

                // 每 5 epoch 验证一次
                var valAcc = 0f
                if (ep % 5 == 4 || ep == phase.epochs - 1) {
                    valAcc = validate(valSet)
                    if (valAcc > bestValAcc) bestValAcc = valAcc
                }

                onProgress?.invoke(DistillStatus(
                    phase.id, phase.name, ep + 1, phase.epochs, epLoss, epAcc, phaseSamples.size
                ))
            }
        }

        val finalValAcc = validate(valSet)
        val rules = extractRules()

        return TrainResult(
            finalLoss = 0f,  // 不跨阶段追踪
            finalAccuracy = validate(trainSet),
            validationAccuracy = finalValAcc,
            rulesExtracted = rules.size
        )
    }

    private fun validate(samples: List<DistilledSample>): Float {
        if (samples.isEmpty()) return 0f
        var correct = 0
        for (s in samples) {
            val pred = predict(s.features)
            if (pred == s.behaviorLabel) correct++
        }
        return correct.toFloat() / samples.size
    }

    // ==================== 规则提取 ====================

    /**
     * 对全部 64 卦提取决策规则：MLP 预测 vs hex_weights 预测的对比
     */
    fun extractRules(): List<DecisionRule> {
        val rules = mutableListOf<DecisionRule>()

        // 为每个卦象生成典型状态
        val typicalStates = Array(64) { hex ->
            // 从 hex 的 6爻 反推典型状态
            val l1 = if ((hex and 1) != 0) 7f else 3f      // atp
            val l2 = if ((hex and 2) != 0) 0.1f else 0.5f   // damage
            val l3 = if ((hex and 4) != 0) 0.5f else 3f     // cortisol
            val l4 = if ((hex and 8) != 0) 2f else 0f       // children
            val l5 = if ((hex and 16) != 0) 2f else 0.5f    // dopamine
            val l6 = if ((hex and 32) != 0) 50f else 150f   // age (不参与输入但记录)
            floatArrayOf(l1, 10f, l2, l3, l5, l4)           // atp,glucose,damage,cortisol,dopamine,children
        }

        for (hex in 0 until 64) {
            val features = typicalStates[hex]
            val (mlpLine, mlpConf) = predictWithConfidence(features)
            val hexName = Hexagram.NAMES[hex]
            val hexWeightLine = Hexagram.BASE_BIAS.indices.maxByOrNull { Hexagram.BASE_BIAS[it] } ?: 5

            rules.add(DecisionRule(
                hexagram = hex,
                guaName = hexName,
                mlpLine = mlpLine,
                mlpBehavior = BEHAVIORS[mlpLine],
                mlpConfidence = mlpConf,
                hexWeightsLine = hexWeightLine,
                hexWeightsBehavior = BEHAVIORS[hexWeightLine],
                agree = mlpLine == hexWeightLine
            ))
        }

        return rules
    }

    // ==================== 影子模式 ====================

    /**
     * 影子预测：返回 MLP 选择的行为线 + 置信度
     * 在 CellEngine.tick() 中与 hex_weights 的 softmax 选出的变爻比较
     */
    fun shadowPredict(atp: Float, glucose: Float, damage: Float,
                      cortisol: Float, dopamine: Float, children: Int): Pair<Int, Float> {
        return predictWithConfidence(floatArrayOf(atp, glucose, damage, cortisol, dopamine, children.toFloat()))
    }

    // ==================== 权重持久化 ====================
    fun saveWeights(file: File) {
        file.outputStream().use { out ->
            val bo = ByteOrder.LITTLE_ENDIAN
            out.write(ByteBuffer.allocate(4).order(bo).putInt(6).array())
            writeTensor(out, w1, intArrayOf(HIDDEN1, INPUT_DIM), bo)
            writeTensor1D(out, b1, bo)
            writeTensor(out, w2, intArrayOf(HIDDEN2, HIDDEN1), bo)
            writeTensor1D(out, b2, bo)
            writeTensor(out, w3, intArrayOf(OUTPUT_DIM, HIDDEN2), bo)
            writeTensor1D(out, b3, bo)
        }
    }

    fun loadWeights(file: File): Boolean {
        if (!file.exists() || file.length() < 4000) return false
        try {
            val buf = ByteBuffer.wrap(file.readBytes()).order(ByteOrder.LITTLE_ENDIAN)
            val nTensors = buf.int
            for (t in 0 until min(nTensors, 6)) {
                val ndim = buf.int
                val dims = IntArray(ndim) { buf.int }
                val size = dims.fold(1) { a, b -> a * b }
                val data = FloatArray(size) { buf.float }
                when (t) {
                    0 -> { for (r in 0 until dims[0]) w1[r] = data.copyOfRange(r * dims[1], (r + 1) * dims[1]) }
                    1 -> b1 = data
                    2 -> { for (r in 0 until dims[0]) w2[r] = data.copyOfRange(r * dims[1], (r + 1) * dims[1]) }
                    3 -> b2 = data
                    4 -> { for (r in 0 until dims[0]) w3[r] = data.copyOfRange(r * dims[1], (r + 1) * dims[1]) }
                    5 -> b3 = data
                }
            }
            return true
        } catch (e: Exception) { return false }
    }

    private fun writeTensor(os: java.io.OutputStream, arr: Array<FloatArray>, dims: IntArray, bo: ByteOrder) {
        os.write(ByteBuffer.allocate(4).order(bo).putInt(dims.size).array())
        for (d in dims) os.write(ByteBuffer.allocate(4).order(bo).putInt(d).array())
        for (row in arr) for (f in row) os.write(ByteBuffer.allocate(4).order(bo).putFloat(f).array())
    }

    private fun writeTensor1D(os: java.io.OutputStream, arr: FloatArray, bo: ByteOrder) {
        os.write(ByteBuffer.allocate(4).order(bo).putInt(1).array())
        os.write(ByteBuffer.allocate(4).order(bo).putInt(arr.size).array())
        for (f in arr) os.write(ByteBuffer.allocate(4).order(bo).putFloat(f).array())
    }

    // ==================== 简易 JSON 解析 ====================
    private fun parseSimpleJson(line: String): Map<String, String> {
        val map = mutableMapOf<String, String>()
        var i = 0
        while (i < line.length) {
            if (line[i] == '"') {
                val keyEnd = line.indexOf('"', i + 1)
                if (keyEnd < 0) break
                val key = line.substring(i + 1, keyEnd)
                i = keyEnd + 1
                while (i < line.length && line[i] != ':') i++
                i++ // skip ':'
                while (i < line.length && line[i] == ' ') i++
                if (i >= line.length) break
                if (line[i] == '"') {
                    val valEnd = line.indexOf('"', i + 1)
                    if (valEnd < 0) break
                    map[key] = line.substring(i + 1, valEnd)
                    i = valEnd + 1
                } else {
                    // number or boolean
                    val start = i
                    while (i < line.length && line[i] != ',' && line[i] != '}' && line[i] != ' ') i++
                    map[key] = line.substring(start, i)
                }
            } else {
                i++
            }
        }
        return map
    }
}
