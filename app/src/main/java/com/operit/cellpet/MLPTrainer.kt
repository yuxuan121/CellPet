package com.operit.cellpet

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.*

/**
 * 纯 Kotlin MLP (12→64→64→8), 与 Python 训练脚本架构一致。
 * 只做推理，不依赖 ONNX Runtime native 库。
 */
class MLPTrainer {
    var w1: Array<FloatArray> = emptyArray(); var b1 = FloatArray(0)
    var w2: Array<FloatArray> = emptyArray(); var b2 = FloatArray(0)
    var w3: Array<FloatArray> = emptyArray(); var b3 = FloatArray(0)
    val inputDim = 12; val hidden = 64; val output = 8

    constructor() {
        val rng = kotlin.random.Random(42)
        w1 = Array(hidden) { FloatArray(inputDim) { rng.nextFloat() * 0.1f - 0.05f } }
        b1 = FloatArray(hidden)
        w2 = Array(hidden) { FloatArray(hidden) { rng.nextFloat() * 0.1f - 0.05f } }
        b2 = FloatArray(hidden)
        w3 = Array(output) { FloatArray(hidden) { rng.nextFloat() * 0.1f - 0.05f } }
        b3 = FloatArray(output)
    }

    constructor(weightsFile: File) : this() {
        if (weightsFile.exists() && weightsFile.length() >= 22048) {
            val buf = ByteBuffer.wrap(weightsFile.readBytes()).order(ByteOrder.LITTLE_ENDIAN)
            val nTensors = buf.int
            for (t in 0 until nTensors) {
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
        }
    }

    /** Softmax前向 */
    fun forward(xNorm: FloatArray): FloatArray {
        val h1 = FloatArray(hidden) { i -> var s = b1[i]; for (j in 0 until inputDim) s += w1[i][j] * xNorm[j]; max(0f, s) }
        val h2 = FloatArray(hidden) { i -> var s = b2[i]; for (j in 0 until hidden) s += w2[i][j] * h1[j]; max(0f, s) }
        val logits = FloatArray(output) { i -> var s = b3[i]; for (j in 0 until hidden) s += w3[i][j] * h2[j]; s }
        val mx = logits.maxOrNull()!!
        val exp = FloatArray(output) { exp(logits[it] - mx) }
        val s = exp.sum()
        return FloatArray(output) { exp[it] / s }
    }

    fun normalize(raw: FloatArray) = FloatArray(inputDim) { i -> raw[i] } // passthrough for now

    fun predict(features: FloatArray): String {
        val p = forward(normalize(features))
        val idx = p.indices.maxByOrNull { p[it] }!!
        return arrayOf("divide","repair","apoptose","immune_alert","store","wait","idle","senescence")[idx]
    }
}
