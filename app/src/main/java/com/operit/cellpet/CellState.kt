package com.operit.cellpet

data class CellState(
    // 核心生理指标
    var atp: Float = 6.5f,
    var glucose: Float = 10f,
    var damage: Float = 0.1f,
    var children: Int = 0,
    var generation: Int = 1,

    // 激素/神经递质
    var cortisol: Float = 0.5f,
    var dopamine: Float = 1.5f,
    var serotonin: Float = 1.0f,

    // 辅助指标
    var nutrient: Float = 2.0f,
    var divideCount: Int = 0,

    // 状态
    var behavior: String = "wait",
    var alive: Boolean = true,
    var age: Long = 0L,

    // ===== 易经卦象（派生字段） =====
    /** 当前卦象索引 0~63 */
    var hexagram: Int = 63,  // 初始为乾卦䷀（全阳：新生细胞）
    /** 上次变爻 0~5，-1表示无变 */
    var changingLine: Int = -1,
    /** 之卦（变后卦象） */
    var targetHexagram: Int = 63,
    /** 上次卦变是否被用户认可（喂食/安抚） */
    var lastChangeAcked: Boolean = false
) {
    companion object {
        val BEHAVIOR_NAMES = mapOf(
            "divide" to "分裂", "repair" to "修复", "apoptose" to "凋亡",
            "store" to "储存", "wait" to "等待"
        )
    }

    fun behaviorName(): String = BEHAVIOR_NAMES[behavior] ?: behavior

    /** 计算当前卦象并更新 hexagram 字段 */
    fun computeHexagram() {
        hexagram = Hexagram.computeHexagram(atp, damage, cortisol, children, dopamine, age)
    }

    /** 获取某个爻的阴阳（0=阴，1=阳） */
    fun lineYang(lineIndex: Int): Boolean {
        val mask = 1 shl lineIndex
        return (hexagram and mask) != 0
    }
}