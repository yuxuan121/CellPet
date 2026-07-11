package com.operit.cellpet

import android.os.Bundle
import android.util.Log
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import ai.onnxruntime.*

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val tv = TextView(this)
        tv.textSize = 14f
        val sb = StringBuilder()
        sb.appendLine("CellPet v1.0.2 — ONNX诊断")

        try {
            Log.d("CellPet", "Step 1: getEnvironment...")
            sb.appendLine("1. OrtEnvironment.getEnvironment()...")
            val env = OrtEnvironment.getEnvironment()
            sb.appendLine("   ✅ OK")

            Log.d("CellPet", "Step 2: read model...")
            sb.appendLine("2. Reading model from assets...")
            val bytes = assets.open("cell_decision_model.onnx").readBytes()
            sb.appendLine("   ✅ ${bytes.size} bytes")

            Log.d("CellPet", "Step 3: createSession...")
            sb.appendLine("3. Creating ONNX session...")
            val session = env.createSession(bytes)
            sb.appendLine("   ✅ Session created")

            Log.d("CellPet", "Step 4: test inference...")
            sb.appendLine("4. Running test inference...")
            val input = FloatArray(12) { 0.5f }
            val tensor = OnnxTensor.createTensor(env, java.nio.FloatBuffer.wrap(input), longArrayOf(1, 12))
            val result = session.run(mapOf("input" to tensor))
            val output = result[0].value as Array<FloatArray>
            sb.appendLine("   ✅ Output: ${output[0].size} classes")
            sb.appendLine("   Probs: ${output[0].take(4).joinToString { "%.3f".format(it) }}...")

            result.close()
            session.close()
            sb.appendLine("\n🎉 ONNX 全部正常！")
        } catch (e: Throwable) {
            Log.e("CellPet", "CRASH: ${e.javaClass.name}: ${e.message}", e)
            sb.appendLine("\n❌ ${e.javaClass.simpleName}")
            sb.appendLine("   ${e.message}")
        }

        setContentView(tv)
    }
}
