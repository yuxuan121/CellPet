package com.operit.cellpet

import android.os.Bundle
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val tv = TextView(this)
        tv.text = "Hello CellPet v1.0.1\nNo ONNX - minimal test"
        tv.textSize = 18f
        setContentView(tv)
    }
}
