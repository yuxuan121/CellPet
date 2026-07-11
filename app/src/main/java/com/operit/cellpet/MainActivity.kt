package com.operit.cellpet

import android.os.Bundle
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val tv = TextView(this)
        tv.textSize = 14f
        setContentView(tv)

        tv.text = "Step 1..."

        try {
            tv.text = "Step 2: MLPTrainer()..."
            val t = MLPTrainer()
            tv.text = "Step 3: OK, forward..."
            val r = t.forward(FloatArray(12) { 0.5f })
            tv.text = "Step 4: OK! Output=${r.joinToString { "%.2f".format(it) }}"
        } catch (e: Throwable) {
            tv.text = "CRASH: ${e.javaClass.simpleName}\n${e.message}"
            e.printStackTrace()
        }
    }
}
