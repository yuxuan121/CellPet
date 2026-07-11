package com.operit.cellpet

import android.content.Intent
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.widget.Button
import android.widget.ProgressBar
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity

class MainActivity : AppCompatActivity() {
    private lateinit var engine: CellEngine
    private val handler = Handler(Looper.getMainLooper())
    private var running = false

    private lateinit var tvBehavior: TextView
    private lateinit var tvAge: TextView
    private lateinit var tvChildren: TextView
    private lateinit var tvStatus: TextView
    private lateinit var pbAtp: ProgressBar
    private lateinit var pbGlucose: ProgressBar
    private lateinit var pbDamage: ProgressBar

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        tvBehavior = findViewById(R.id.tvBehavior)
        tvAge = findViewById(R.id.tvAge)
        tvChildren = findViewById(R.id.tvChildren)
        tvStatus = findViewById(R.id.tvStatus)
        pbAtp = findViewById(R.id.pbAtp)
        pbGlucose = findViewById(R.id.pbGlucose)
        pbDamage = findViewById(R.id.pbDamage)
        val btnFeed: Button = findViewById(R.id.btnFeed)
        val btnSoothe: Button = findViewById(R.id.btnSoothe)

        engine = CellEngine.getInstance(this)
        if (!engine.state.alive) {
            tvStatus.text = "Failed to load ONNX model"
            return
        }

        btnFeed.setOnClickListener { engine.feed(); updateUI() }
        btnSoothe.setOnClickListener { engine.soothe(); updateUI() }

        running = true
        startService(Intent(this, CellService::class.java))
        startMainLoop()
    }

    private fun startMainLoop() {
        handler.postDelayed(object : Runnable {
            override fun run() {
                if (!running) return
                engine.tick()
                updateUI()
                handler.postDelayed(this, 3000)
            }
        }, 3000)
    }

    private fun updateUI() {
        val s = engine.state
        if (!s.alive) {
            tvBehavior.text = "Died"
            tvStatus.text = "Lived " + s.age + " ticks, " + s.children + " children"
            running = false
            return
        }
        tvBehavior.text = s.behaviorName()
        tvAge.text = "Age: " + s.age
        tvChildren.text = "Children: " + s.children + " (Gen " + s.generation + ")"
        pbAtp.progress = (s.atp / 10f * 100).toInt()
        pbGlucose.progress = (s.glucose / 20f * 100).toInt()
        pbDamage.progress = (s.damage * 100).toInt()
        tvStatus.text = "ATP: " + "%.1f".format(s.atp) + " | Damage: " + "%.0f".format(s.damage * 100) + "%"
    }

    override fun onDestroy() {
        running = false
        engine.close()
        super.onDestroy()
    }
}
