package com.operit.cellpet

import android.content.Context
import java.io.File
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
    private val trainer: MLPTrainer

    init {
        // Copy weights from assets
        val wf = File(context.filesDir, "cell_weights.bin")
        if (!wf.exists()) {
            context.assets.open("cell_weights.bin").use { inp ->
                wf.outputStream().use { out -> inp.copyTo(out) }
            }
        }
        trainer = MLPTrainer(wf)
    }

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

        // MLP inference
        val input = state.toMLPInput()
        val probs = trainer.forward(trainer.normalize(input))
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

        state.atp = state.atp.coerceIn(0f, 10f)
        state.glucose = state.glucose.coerceIn(0f, 20f)
        state.damage = state.damage.coerceIn(0f, 1f)
        return state
    }

    fun feed() { state.glucose += 3f; state.nutrient += 1f }
    fun soothe() { state.cortisol = (state.cortisol - 1f).coerceAtLeast(0f); state.dopamine += 0.5f }
    fun reset() { state = CellState() }
    fun close() {}
}