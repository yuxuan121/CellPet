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
    private lateinit var tvTrainStatus: TextView
    private lateinit var btnTrain: Button
    private lateinit var btnReset: Button
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
        tvTrainStatus = findViewById(R.id.tvTrainStatus)
        btnTrain = findViewById(R.id.btnTrain)
        btnReset = findViewById(R.id.btnReset)
        pbAtp = findViewById(R.id.pbAtp)
        pbGlucose = findViewById(R.id.pbGlucose)
        pbDamage = findViewById(R.id.pbDamage)
        val btnFeed: Button = findViewById(R.id.btnFeed)
        val btnSoothe: Button = findViewById(R.id.btnSoothe)

        engine = CellEngine.getInstance(this)
        if (!engine.state.alive) {
            tvStatus.text = "模型加载失败"
            return
        }

        btnFeed.setOnClickListener { engine.feed(); updateUI() }
        btnSoothe.setOnClickListener { engine.soothe(); updateUI() }
        btnTrain.setOnClickListener { startTraining() }
        btnReset.setOnClickListener { engine.reset(); running = true; updateUI() }

        running = true
        try { startService(Intent(this, CellService::class.java)) } catch(e: Exception) {}
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
            tvBehavior.text = "已凋亡"
            tvStatus.text = "存活: " + s.age + " 轮 | 子代: " + s.children
            running = false
            return
        }
        tvBehavior.text = s.behaviorName()
        tvAge.text = "年龄: " + s.age
        tvChildren.text = "子代: " + s.children + " (第" + s.generation + "代)"
        pbAtp.progress = (s.atp / 10f * 100).toInt()
        pbGlucose.progress = (s.glucose / 20f * 100).toInt()
        pbDamage.progress = (s.damage * 100).toInt()
        tvStatus.text = "ATP: " + "%.1f".format(s.atp) + "/10 | 葡萄糖: " + "%.1f".format(s.glucose) + "/20 | 损伤: " + "%.0f".format(s.damage * 100) + "%"
        tvTrainStatus.text = "样本: " + engine.getSampleCount() + " | " + engine.trainStatus.message
    }

    private fun startTraining() {
        if (engine.trainStatus.running) {
            tvTrainStatus.text = "训练进行中,请稍候..."
            return
        }
        btnTrain.isEnabled = false
        tvTrainStatus.text = "准备训练..."

        Thread {
            engine.selfTrain(epochs = 30, lr = 0.01f, batchSize = 16) { status ->
                runOnUiThread {
                    tvTrainStatus.text = "样本: " + engine.getSampleCount() +
                        " | [" + status.currentEpoch + "/" + status.totalEpochs + "] " +
                        "Loss=" + "%.4f".format(status.loss) +
                        " Acc=" + "%.1f".format(status.acc * 100) + "%"
                }
            }
            runOnUiThread {
                btnTrain.isEnabled = true
                tvTrainStatus.text = "样本: " + engine.getSampleCount() + " | " + engine.trainStatus.message
            }
        }.start()
    }

    override fun onDestroy() {
        running = false
        engine.close()
        super.onDestroy()
    }
}
