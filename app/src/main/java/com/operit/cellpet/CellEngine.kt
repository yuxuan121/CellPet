package com.operit.cellpet

import android.content.Context
import android.util.Log
import ai.onnxruntime.*
import java.nio.FloatBuffer
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

    private var session: OrtSession? = null
    private var env: OrtEnvironment? = null
    var state = CellState()

    init {
        try {
            Log.d("CellPet", "Loading ONNX model...")
            env = OrtEnvironment.getEnvironment()
            val modelBytes = context.assets.open("cell_decision_model.onnx").readBytes()
            Log.d("CellPet", "Model size: ${modelBytes.size} bytes")
            session = env?.createSession(modelBytes)
            Log.d("CellPet", "ONNX session created OK")
        } catch (e: Throwable) {
            Log.e("CellPet", "Failed to load ONNX: ${e.javaClass.simpleName}: ${e.message}", e)
            state.alive = false
        }
    }

    fun tick(stimulus: String = "", isCharging: Boolean = false): CellState {
        if (!state.alive || session == null) return state
        state.age++

        state.atp -= 0.05f + Random.nextFloat() * 0.1f
        state.glucose -= 0.03f
        state.damage += 0.002f
        state.nutrient = (state.nutrient - 0.01f).coerceAtLeast(0f)

        when (stimulus) {
            "shake" -> { state.cortisol += 0.5f; state.damage += 0.02f }
            "dark" -> { state.serotonin += 0.2f }
            "quiet" -> { state.dopamine += 0.1f }
        }
        if (isCharging) { state.glucose += 0.5f; state.nutrient += 0.2f }

        state.cortisol = (state.cortisol - 0.1f).coerceAtLeast(0f)
        state.dopamine = (state.dopamine - 0.02f).coerceAtLeast(0f)
        state.serotonin = (state.serotonin - 0.01f).coerceAtLeast(0f)

        if (state.atp < 1.0f) state.damage += 0.05f
        if (state.glucose < 0.5f) state.damage += 0.03f

        try {
            val input = state.toInputArray()
            val tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(input), longArrayOf(1, 12))
            val result = session!!.run(mapOf("input" to tensor))
            val output = result[0].value as Array<FloatArray>
            val probs = output[0]
            val maxIdx = probs.indices.maxByOrNull { probs[it] } ?: 5
            state.behavior = CellState.BEHAVIORS[maxIdx]
            result.close()
        } catch (e: Exception) {
            state.behavior = when {
                state.damage > 0.8f -> "apoptose"
                state.atp < 2f -> "wait"
                state.damage > 0.3f -> "repair"
                state.atp > 7f && state.damage < 0.2f -> "divide"
                else -> "idle"
            }
        }

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
    fun close() { session?.close(); env?.close() }
}
