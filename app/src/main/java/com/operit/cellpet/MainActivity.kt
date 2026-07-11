package com.operit.cellpet

import android.os.Bundle
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val tv = TextView(this)
        tv.textSize = 13f
        val sb = StringBuilder()
        sb.appendLine("CellPet v1.0.3 — 纯Kotlin推理")
        
        try {
            val trainer = MLPTrainer()
            sb.appendLine("1. MLPTrainer init ✅")
            
            // Load weights from binary
            val wf = java.io.File(filesDir, "cell_weights.bin")
            if (!wf.exists()) {
                assets.open("cell_weights.bin").use { inp ->
                    wf.outputStream().use { out -> inp.copyTo(out) }
                }
            }
            val trainer2 = MLPTrainer(wf)
            sb.appendLine("2. Weights loaded ✅ (${wf.length()} bytes)")
            
            // Test inference
            val feat = floatArrayOf(6.5f, 0.1f, 0.5f, 1.5f, 0.01f, 0.0f, 0.0f, 0f, 0.08f, 10f, 0.7f, 0.01f)
            val pred = trainer2.predict(feat)
            sb.appendLine("3. Predict: $pred ✅")
            
            val probs = trainer2.forward(trainer2.normalize(feat))
            sb.appendLine("4. Probs: ${probs.take(4).joinToString { "%.3f".format(it) }}... ✅")
            
            sb.appendLine("\n🎉 纯Kotlin推理成功！")
            sb.appendLine("   不依赖onnxruntime native库")
        } catch (e: Throwable) {
            sb.appendLine("\n❌ ${e.javaClass.simpleName}: ${e.message}")
        }
        
        setContentView(tv)
    }
}
