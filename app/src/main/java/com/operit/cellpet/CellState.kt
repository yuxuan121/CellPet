package com.operit.cellpet

data class CellState(
    var atp: Float = 6.5f,
    var glucose: Float = 10f,
    var damage: Float = 0.1f,
    var divideCount: Int = 0,
    var cortisol: Float = 0.5f,
    var estrogen: Float = 1.0f,
    var dopamine: Float = 1.5f,
    var serotonin: Float = 1.0f,
    var nutrient: Float = 2.0f,
    var damageSig: Float = 0.01f,
    var hormone: Float = 1.0f,
    var children: Int = 0,
    var generation: Int = 1,
    var behavior: String = "idle",
    var alive: Boolean = true,
    var age: Long = 0L
) {
    companion object {
        val BEHAVIORS = arrayOf("divide", "repair", "apoptose", "immune_alert",
            "store", "wait", "idle", "senescence")
        val BEHAVIOR_NAMES = mapOf(
            "divide" to "分裂", "repair" to "修复", "apoptose" to "凋亡",
            "immune_alert" to "免疫", "store" to "储存", "wait" to "等待",
            "idle" to "空闲", "senescence" to "衰老"
        )
    }

    fun behaviorName(): String = BEHAVIOR_NAMES[behavior] ?: behavior

    fun toInputArray(): FloatArray = floatArrayOf(
        atp / 10f, glucose / 20f, damage, divideCount / 50f,
        cortisol / 10f, estrogen / 8f, dopamine / 5f,
        serotonin / 3f, nutrient / 5f, damageSig,
        hormone / 2f,
        (atp / 10f).coerceIn(0f, 1f)
    )
}
