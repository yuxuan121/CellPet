package com.operit.cellpet

import android.os.Bundle
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import kotlin.math.*

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val tv = TextView(this)
        tv.textSize = 12f
        setContentView(tv)

        try {
            tv.text = "Loading weights..."

            // Copy weights from assets to internal storage
            val wf = java.io.File(filesDir, "cell_weights.bin")
            if (!wf.exists()) {
                assets.open("cell_weights.bin").use { inp ->
                    wf.outputStream().use { out -> inp.copyTo(out) }
                }
            }
            tv.text = "Weights: ${wf.length()} bytes\nLoading..."

            val trainer = MLPTrainer(wf)
            tv.text = "Weights loaded\nRunning inference..."

            // Test scenarios
            val tests = mapOf(
                "Healthy" to floatArrayOf(7.5f, 0.05f, 0.5f, 1.5f, 0.01f, 0.0f, 0.0f, 5f, 0.08f, 8f, 0.5f, 0.01f),
                "Damaged" to floatArrayOf(4.0f, 0.45f, 3.0f, 0.8f, 0.02f, 0.0f, 0.0f, 15f, 0.08f, 5f, 1.2f, 0.01f),
                "Dying" to floatArrayOf(0.8f, 0.85f, 6.0f, 0.3f, 0.05f, 0.0f, 0.0f, 35f, 0.08f, 1f, 1.8f, 0.02f),
            )

            val sb = StringBuilder()
            sb.appendLine("CellPet v1.0.6 — Trained Inference")
            sb.appendLine()

            for ((name, feat) in tests) {
                val probs = trainer.forward(trainer.normalize(feat))
                val pred = trainer.predict(feat)
                sb.appendLine("$name: $pred")
                sb.appendLine("  ${probs.joinToString { "%.3f".format(it) }}")
                sb.appendLine()
            }

            tv.text = sb.toString()
        } catch (e: Throwable) {
            tv.text = "CRASH: ${e.javaClass.simpleName}\n${e.message}\n\n${e.stackTraceToString().take(500)}"
        }
    }
}
