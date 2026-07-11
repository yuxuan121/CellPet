package com.operit.cellpet

import android.content.Context
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.File
import kotlin.random.Random
import kotlin.math.*

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
    private val ctx = context.applicationContext

    // ==================== 卦变权重表 [64卦][6爻] ====================
    // 权重越大，该卦象下选择该变爻的概率越高
    var weights: Array<FloatArray> = Array(64) { Hexagram.BASE_BIAS.copyOf() }
    private val weightFile: File get() = File(ctx.filesDir, "hex_weights.bin")

    init {
        loadWeights()
        // 初始卦象
        state.computeHexagram()
    }

    // ==================== 权重持久化 ====================

    private fun loadWeights() {
        if (!weightFile.exists()) return
        try {
            DataInputStream(weightFile.inputStream().buffered()).use { dis ->
                for (g in 0 until 64) {
                    for (l in 0 until 6) {
                        weights[g][l] = dis.readFloat()
                    }
                }
            }
        } catch (e: Exception) { /* 保留默认偏置 */ }
    }

    private fun saveWeights() {
        try {
            DataOutputStream(weightFile.outputStream().buffered()).use { dos ->
                for (g in 0 until 64) {
                    for (l in 0 until 6) {
                        dos.writeFloat(weights[g][l])
                    }
                }
            }
        } catch (e: Exception) { /* 静默失败 */ }
    }

    // ==================== 代谢与卦变决策 ====================

    fun tick(): CellState {
        if (!state.alive) return state

        state.age++

        // ---- 代谢 ----
        // 葡萄糖→ATP 呼吸转化
        if (state.glucose > 0.5f) {
            val converted = min(state.glucose * 0.05f, 0.3f)
            state.glucose -= converted
            state.atp += converted * 6f
        }
        state.atp -= 0.05f + Random.nextFloat() * 0.1f
        state.glucose -= 0.03f
        state.damage += 0.002f
        state.nutrient = (state.nutrient - 0.01f).coerceAtLeast(0f)
        state.cortisol = (state.cortisol - 0.1f).coerceAtLeast(0f)
        state.dopamine = (state.dopamine - 0.02f).coerceAtLeast(0f)
        state.serotonin = (state.serotonin - 0.01f).coerceAtLeast(0f)

        // 低能量/低血糖 → 损伤加速
        if (state.atp < 1.0f) state.damage += 0.05f
        if (state.glucose < 0.5f) state.damage += 0.03f

        // ---- 计算当前卦象 ----
        state.computeHexagram()
        val hex = state.hexagram

        // ---- 确定变化概率（压力越大，越可能寻求变化） ----
        var changeProb = 0.2f
        if (state.damage > 0.5f) changeProb += 0.2f
        if (state.cortisol > 3f) changeProb += 0.15f
        if (state.atp < 2f) changeProb += 0.15f
        changeProb = changeProb.coerceIn(0.1f, 0.8f)

        // ---- 卦变决策 ----
        if (Random.nextFloat() < changeProb) {
            // softmax 选择变爻
            val w = weights[hex]
            val temperature = 0.5f
            val maxW = w.maxOrNull()!!
            val expW = FloatArray(6) { exp((w[it] - maxW) / temperature) }
            val sumExp = expW.sum()
            val probs = FloatArray(6) { expW[it] / sumExp }

            // 按概率采样
            val r = Random.nextFloat()
            var cum = 0f
            var chosen = 5  // 默认上爻（最安全的选择）
            for (i in 0 until 6) {
                cum += probs[i]
                if (r <= cum) { chosen = i; break }
            }

            val lineYangBefore = state.lineYang(chosen)
            state.changingLine = chosen
            state.targetHexagram = hex xor (1 shl chosen)

            // 变爻 → 行为
            state.behavior = Hexagram.LINE_BEHAVIORS[chosen]

            // ---- 执行行为 ----
            when (state.behavior) {
                "divide" -> {
                    if (state.atp > 3f && state.damage < 0.3f) {
                        state.children++; state.divideCount++; state.atp -= 3f
                    } else {
                        state.behavior = "wait"  // 条件不满足，改为等待
                        state.changingLine = -1
                    }
                }
                "repair" -> {
                    if (state.atp > 2f) {
                        state.damage = (state.damage * 0.5f).coerceAtLeast(0f); state.atp -= 2f
                    } else {
                        state.behavior = "wait"; state.changingLine = -1
                    }
                }
                "apoptose" -> {
                    state.alive = false
                }
                "store" -> {
                    state.glucose += 1f; state.atp -= 0.5f
                }
                "wait" -> {
                    state.atp += 0.1f  // 等待时微量恢复
                }
            }
        } else {
            // 不变：等待
            state.behavior = "wait"
            state.changingLine = -1
            state.atp += 0.05f  // 静息微量恢复
        }

        // 上轮卦变未获反馈 → 轻微遗忘
        if (state.changingLine >= 0 && !state.lastChangeAcked) {
            val oldLine = state.changingLine
            // 下一轮再处理（这轮刚做完决策，还没机会被 ack）
            // 实际遗忘在上轮没被 ack 时发生
        }
        state.lastChangeAcked = false

        // 钳位
        state.atp = state.atp.coerceIn(0f, 10f)
        state.glucose = state.glucose.coerceIn(0f, 20f)
        state.damage = state.damage.coerceIn(0f, 1f)
        return state
    }

    // ==================== 用户交互 ====================

    fun feed() {
        state.glucose += 3f; state.nutrient += 1f; state.atp += 2f
        reinforce(1.0f)  // 认可当前卦变
        state.lastChangeAcked = true
    }

    fun soothe() {
        state.cortisol = (state.cortisol - 1f).coerceAtLeast(0f); state.dopamine += 0.5f
        reinforce(0.5f)  // 温和认可
        state.lastChangeAcked = true
    }

    /** 强化学习：用户交互 → 当前卦变权重增加 */
    private fun reinforce(delta: Float) {
        val hex = state.hexagram
        val line = state.changingLine
        if (line < 0 || line > 5) return
        // 更新权重
        weights[hex][line] += delta
        // 衰减其他变爻（winner-take-more）
        for (i in 0 until 6) {
            if (i != line) weights[hex][i] *= 0.95f
        }
        saveWeights()
    }

    // ==================== 重置 ====================

    fun reset() {
        state = CellState()
        state.computeHexagram()
    }

    fun close() {}
}