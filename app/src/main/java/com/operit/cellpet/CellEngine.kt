package com.operit.cellpet

import android.content.Context
import java.io.File
import java.text.SimpleDateFormat
import java.util.*
import kotlin.random.Random

class CellEngine private constructor(context: Context) {
    companion object {
        private var INSTANCE: CellEngine? = null
        fun getInstance(context: Context): CellEngine {
            return INSTANCE ?: synchronized(this) {
                INSTANCE ?: CellEngine(context.applicationContext).also { INSTANCE = it }
            }
        }
    }

    var state = CellState()
    val trainer: MLPTrainer
    private val ctx = context.applicationContext

    // ---- 训练数据 ----
    data class Sample(val features: FloatArray, val target: Int, val timestamp: Long = System.currentTimeMillis())
    private val samples = mutableListOf<Sample>()
    private var userActionLast3: MutableList<Int> = mutableListOf(0, 0, 0)  // 0=none,1=feed,2=soothe

    // ---- 训练状态 ----
    data class TrainStatus(
        var running: Boolean = false,
        var currentEpoch: Int = 0,
        var totalEpochs: Int = 0,
        var loss: Float = 0f,
        var acc: Float = 0f,
        var message: String = "就绪"
    )
    val trainStatus = TrainStatus()

    init {
        val wf = File(ctx.filesDir, "cell_weights.bin")
        if (!wf.exists()) {
            ctx.assets.open("cell_weights.bin").use { inp ->
                wf.outputStream().use { out -> inp.copyTo(out) }
            }
        }
        trainer = MLPTrainer(wf)
    }

    // ==================== 训练数据采集 ====================

    /** 每个 tick 记录当前状态+模型预测行为 */
    fun tick(): CellState {
        if (!state.alive) return state

        state.age++

        // Metabolism
        state.atp -= 0.05f + Random.nextFloat() * 0.1f
        state.glucose -= 0.03f
        state.damage += 0.002f
        state.nutrient = (state.nutrient - 0.01f).coerceAtLeast(0f)
        state.cortisol = (state.cortisol - 0.1f).coerceAtLeast(0f)
        state.dopamine = (state.dopamine - 0.02f).coerceAtLeast(0f)
        state.serotonin = (state.serotonin - 0.01f).coerceAtLeast(0f)

        if (state.atp < 1.0f) state.damage += 0.05f
        if (state.glucose < 0.5f) state.damage += 0.03f

        // Current state features
        val features = state.toMLPInput()

        // User-action-based target: if user recently fed/soothed, override target
        val recentAction = userActionLast3.maxOrNull() ?: 0
        val modelTarget = trainer.forward(trainer.normalize(features)).indices.maxByOrNull { trainer.forward(trainer.normalize(features))[it] } ?: 5

        // Determine target: user action overrides model for recent ticks
        val behaviorIdx: Int
        when (recentAction) {
            1 -> behaviorIdx = 4  // feed → target=store
            2 -> behaviorIdx = 1  // soothe → target=repair
            else -> behaviorIdx = modelTarget  // use model prediction
        }

        // MLP inference (still use model for actual behavior)
        val probs = trainer.forward(trainer.normalize(features))
        val maxIdx = probs.indices.maxByOrNull { probs[it] } ?: 5
        state.behavior = CellState.BEHAVIORS[maxIdx]

        // Execute behavior
        when (state.behavior) {
            "divide" -> { if (state.atp > 3f && state.damage < 0.3f) { state.children++; state.divideCount++; state.atp -= 3f } }
            "repair" -> { if (state.atp > 2f) { state.damage = (state.damage * 0.5f).coerceAtLeast(0f); state.atp -= 2f } }
            "apoptose" -> { state.alive = false }
            "store" -> { state.glucose += 1f; state.atp -= 0.5f }
            "wait" -> { state.atp += 0.1f }
        }

        // Collect sample (target = user-corrected or model's own decision)
        if (samples.size < 5000) {
            samples.add(Sample(features, behaviorIdx))
        }

        // Shift action history
        userActionLast3.removeAt(0)
        userActionLast3.add(0)

        state.atp = state.atp.coerceIn(0f, 10f)
        state.glucose = state.glucose.coerceIn(0f, 20f)
        state.damage = state.damage.coerceIn(0f, 1f)
        return state
    }

    fun feed() {
        state.glucose += 3f; state.nutrient += 1f
        userActionLast3 = mutableListOf(1, 1, 1)  // mark last 3 ticks as feed
    }

    fun soothe() {
        state.cortisol = (state.cortisol - 1f).coerceAtLeast(0f); state.dopamine += 0.5f
        userActionLast3 = mutableListOf(2, 2, 2)  // mark last 3 ticks as soothe
    }

    fun reset() { state = CellState(); samples.clear(); trainStatus.message = "就绪" }
    fun close() {}

    // ==================== 自训练 ====================

    fun getSampleCount(): Int = samples.size

    /** 运行自训练（同步，建议在后台线程调用） */
    fun selfTrain(
        epochs: Int = 30,
        lr: Float = 0.01f,
        batchSize: Int = 16,
        onProgress: ((TrainStatus) -> Unit)? = null
    ) {
        if (samples.size < 50) {
            trainStatus.message = "样本不足（需≥50，当前${samples.size}）"
            trainStatus.running = false
            onProgress?.invoke(trainStatus)
            return
        }

        trainStatus.running = true
        trainStatus.totalEpochs = epochs
        trainStatus.message = "训练中..."

        val trainingData = samples.map { Pair(it.features, it.target) }

        trainer.trainEpochs(trainingData, epochs, lr, batchSize) { ep, loss, acc ->
            trainStatus.currentEpoch = ep
            trainStatus.loss = loss
            trainStatus.acc = acc
            trainStatus.message = "Epoch $ep/$epochs"
            onProgress?.invoke(trainStatus)
        }

        // Save trained weights
        val wf = File(ctx.filesDir, "cell_weights_trained.bin")
        trainer.saveWeights(wf)
        // Also overwrite main weights file for next start
        trainer.saveWeights(File(ctx.filesDir, "cell_weights.bin"))

        trainStatus.running = false
        trainStatus.message = "✅ 训练完成 Loss=${"%.4f".format(trainStatus.loss)} Acc=${"%.1f".format(trainStatus.acc * 100)}%"
        onProgress?.invoke(trainStatus)
    }
}