package com.operit.cellpet

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.*

/**
 * 纯 Kotlin MLP (12→64→64→8), ReLU+Softmax。
 * v2: 新增 mini-batch SGD 反向传播训练能力，用于手机端增量自训练。
 */
class MLPTrainer {
    var w1: Array<FloatArray> = emptyArray(); var b1 = FloatArray(0)
    var w2: Array<FloatArray> = emptyArray(); var b2 = FloatArray(0)
    var w3: Array<FloatArray> = emptyArray(); var b3 = FloatArray(0)
    val inputDim = 12; val hidden = 64; val output = 8

    val mean = floatArrayOf(5.0514f, 0.21768f, 4.0451f, 2.49288f, 0.015057f, 0.007928f,
        0.012156f, 26.948f, 0.0809f, 10.163f, 0.6933f, 0.010007f)
    val std = floatArrayOf(2.8547f, 0.20114f, 2.2987f, 1.4379f, 0.015075f, 0.0079f,
        0.012032f, 15.894f, 0.27271f, 5.6999f, 0.39633f, 0.010041f)

    constructor() {
        val rng = kotlin.random.Random(42)
        w1 = Array(hidden) { FloatArray(inputDim) { rng.nextFloat() * 0.02f - 0.01f } }
        b1 = FloatArray(hidden)
        w2 = Array(hidden) { FloatArray(hidden) { rng.nextFloat() * 0.02f - 0.01f } }
        b2 = FloatArray(hidden)
        w3 = Array(output) { FloatArray(hidden) { rng.nextFloat() * 0.02f - 0.01f } }
        b3 = FloatArray(output)
    }

    constructor(weightsFile: File) : this() {
        loadWeights(weightsFile)
    }

    // ==================== 权重加载 ====================
    fun loadWeights(file: File): Boolean {
        if (!file.exists() || file.length() < 22048) return false
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

    // ==================== 推理 ====================
    fun normalize(raw: FloatArray): FloatArray {
        return FloatArray(inputDim) { i -> (raw[i] - mean[i]) / max(std[i], 1e-8f) }
    }

    fun forward(xNorm: FloatArray): FloatArray {
        val h1 = FloatArray(hidden) { i ->
            var s = b1[i]; for (j in 0 until inputDim) s += w1[i][j] * xNorm[j]; max(0f, s)
        }
        val h2 = FloatArray(hidden) { i ->
            var s = b2[i]; for (j in 0 until hidden) s += w2[i][j] * h1[j]; max(0f, s)
        }
        val logits = FloatArray(output) { i ->
            var s = b3[i]; for (j in 0 until hidden) s += w3[i][j] * h2[j]; s
        }
        val mx = logits.maxOrNull()!!
        val exp = FloatArray(output) { exp(logits[it] - mx) }
        val sumExp = exp.sum()
        return FloatArray(output) { exp[it] / sumExp }
    }

    fun predict(features: FloatArray): String {
        val p = forward(normalize(features))
        val idx = p.indices.maxByOrNull { p[it] }!!
        return arrayOf("divide","repair","apoptose","immune_alert","store","wait","idle","senescence")[idx]
    }

    // ==================== 训练 ====================

    /**
     * 训练一个 mini-batch。返回 (loss, accuracy)。
     */
    fun trainBatch(
        xsNorm: List<FloatArray>, targets: IntArray,
        lr: Float = 0.01f, l2Lambda: Float = 1e-4f, balanceLambda: Float = 0.02f
    ): Pair<Float, Float> {
        val n = xsNorm.size
        if (n == 0) return Pair(0f, 1f)

        // Forward all samples, cache activations
        val a1s = Array(n) { FloatArray(hidden) }
        val a2s = Array(n) { FloatArray(hidden) }
        val probs = Array(n) { FloatArray(output) }

        for (k in 0 until n) {
            val x = xsNorm[k]
            // Layer 1
            for (i in 0 until hidden) {
                var s = b1[i]; for (j in 0 until inputDim) s += w1[i][j] * x[j]
                a1s[k][i] = max(0f, s)
            }
            // Layer2
            for (i in 0 until hidden) {
                var s = b2[i]; for (j in 0 until hidden) s += w2[i][j] * a1s[k][j]
                a2s[k][i] = max(0f, s)
            }
            // Layer 3 + softmax
            var mx = Float.NEGATIVE_INFINITY
            val logits = FloatArray(output) { i ->
                var s = b3[i]; for (j in 0 until hidden) s += w3[i][j] * a2s[k][j]; s
            }
            mx = logits.maxOrNull()!!
            var sumExp = 0f
            for (i in 0 until output) { probs[k][i] = exp(logits[i] - mx); sumExp += probs[k][i] }
            for (i in 0 until output) probs[k][i] /= sumExp
        }

        // Accumulate gradients
        val dW1 = Array(hidden) { FloatArray(inputDim) }
        val db1 = FloatArray(hidden)
        val dW2 = Array(hidden) { FloatArray(hidden) }
        val db2 = FloatArray(hidden)
        val dW3 = Array(output) { FloatArray(hidden) }
        val db3 = FloatArray(output)
        var totalLoss = 0f
        var correct = 0

        for (k in 0 until n) {
            val x = xsNorm[k]
            val a1 = a1s[k]; val a2 = a2s[k]
            val prob = probs[k]; val t = targets[k]

            val pLabel = prob[t].coerceIn(1e-8f, 1f)
            totalLoss -= ln(pLabel)

            val maxIdx = prob.indices.maxByOrNull { prob[it] }!!
            if (maxIdx == t) correct++

            // dz3 = (probs - onehot) / n
            val dz3 = FloatArray(output) { prob[it] / n }
            dz3[t] -= 1f / n

            // Balance: penalize max prob > 0.9
            val pmax = prob.maxOrNull()!!
            val pmaxIdx = prob.indices.maxByOrNull { prob[it] }!!
            if (pmax > 0.9f) {
                for (i in 0 until output) {
                    dz3[i] += balanceLambda * prob[i] * ((if (i == pmaxIdx) 1f else 0f) - prob[pmaxIdx]) * (pmax - 0.9f) / n
                }
            }

            // Layer3 grads
            for (i in 0 until output) {
                for (j in 0 until hidden) dW3[i][j] += dz3[i] * a2[j]
                db3[i] += dz3[i]
            }

            // da2 = dz3 @ W3, then ReLU'
            val dz2 = FloatArray(hidden)
            for (j in 0 until hidden) {
                var da = 0f
                for (i in 0 until output) da += dz3[i] * w3[i][j]
                dz2[j] = if (a2[j] > 0f) da else 0f
            }

            // Layer2 grads
            for (i in 0 until hidden) {
                for (j in 0 until hidden) dW2[i][j] += dz2[i] * a1[j]
                db2[i] += dz2[i]
            }

            // da1 = dz2 @ W2, then ReLU'
            val dz1 = FloatArray(hidden)
            for (j in 0 until hidden) {
                var da = 0f
                for (i in 0 until hidden) da += dz2[i] * w2[i][j]
                dz1[j] = if (a1[j] > 0f) da else 0f
            }

            // Layer1 grads
            for (i in 0 until hidden) {
                for (j in 0 until inputDim) dW1[i][j] += dz1[i] * x[j]
                db1[i] += dz1[i]
            }
        }

        totalLoss /= n
        val acc = correct.toFloat() / n

        // Apply updates with L2
        for (i in 0 until output) {
            for (j in 0 until hidden) w3[i][j] -= lr * (dW3[i][j] + l2Lambda * w3[i][j])
            b3[i] -= lr * db3[i]
        }
        for (i in 0 until hidden) {
            for (j in 0 until hidden) w2[i][j] -= lr * (dW2[i][j] + l2Lambda * w2[i][j])
            b2[i] -= lr * db2[i]
        }
        for (i in 0 until hidden) {
            for (j in 0 until inputDim) w1[i][j] -= lr * (dW1[i][j] + l2Lambda * w1[i][j])
            b1[i] -= lr * db1[i]
        }

        return Pair(totalLoss, acc)
    }

    fun trainEpochs(
        samples: List<Pair<FloatArray, Int>>,
        epochs: Int = 30, lr: Float = 0.01f, batchSize: Int = 16,
        callback: ((Int, Float, Float) -> Unit)? = null
    ) {
        val rng = kotlin.random.Random(System.currentTimeMillis())
        for (ep in 0 until epochs) {
            val shuffled = samples.shuffled(rng)
            var epLoss = 0f; var epAcc = 0f; var cnt = 0
            for (i in shuffled.indices step batchSize) {
                val end = min(i + batchSize, shuffled.size)
                val xsNorm = shuffled.subList(i, end).map { normalize(it.first) }
                val ys = IntArray(end - i) { j -> shuffled[i + j].second }
                val (loss, acc) = trainBatch(xsNorm, ys, lr)
                epLoss += loss * (end - i)
                epAcc += acc * (end - i)
                cnt += end - i
            }
            callback?.invoke(ep + 1, epLoss / cnt, epAcc / cnt)
        }
    }

    // ==================== 权重保存 ====================
    fun saveWeights(file: File) {
        file.outputStream().use { out ->
            val bo = ByteOrder.LITTLE_ENDIAN
            // nTensors
            out.write(ByteBuffer.allocate(4).order(bo).putInt(6).array())
            // W1 (64,12)
            writeTensor(out, w1, intArrayOf(hidden, inputDim), bo)
            // b1 (64)
            writeTensor1D(out, b1, bo)
            // W2 (64,64)
            writeTensor(out, w2, intArrayOf(hidden, hidden), bo)
            // b2 (64)
            writeTensor1D(out, b2, bo)
            // W3 (8,64)
            writeTensor(out, w3, intArrayOf(output, hidden), bo)
            // b3 (8)
            writeTensor1D(out, b3, bo)
        }
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
}