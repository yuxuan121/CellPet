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

    /** Map cell state to 12-D MLP input matching Python training features */
    fun toMLPInput(): FloatArray = floatArrayOf(
        atp,
        damage,
        cortisol,
        dopamine,
        0.015f,                                    // neighbor_damage
        if (damage > 0.3f) 0.02f else 0.008f,      // neighbor_repair
        if (cortisol > 3f) 0.05f else 0.012f,      // immune_alert
        divideCount.toFloat(),                      // division_count
        if (damage > 0.3f || cortisol > 5f) 1f else 0.08f, // is_abnormal
        glucose,                                     // energy_reserve
        (cortisol / 5f).coerceIn(0f, 1f),           // metabolic_stress
        0.01f                                        // neighbor_division
    )
}