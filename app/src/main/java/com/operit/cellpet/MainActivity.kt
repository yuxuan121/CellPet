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
    private lateinit var tvHexagram: TextView
    private lateinit var btnGuanGua: Button
    private lateinit var btnReset: Button
    private lateinit var pbAtp: ProgressBar
    private lateinit var pbGlucose: ProgressBar
    private lateinit var pbDamage: ProgressBar

    private var showDetail = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        tvBehavior = findViewById(R.id.tvBehavior)
        tvAge = findViewById(R.id.tvAge)
        tvChildren = findViewById(R.id.tvChildren)
        tvStatus = findViewById(R.id.tvStatus)
        tvHexagram = findViewById(R.id.tvHexagram)
        btnGuanGua = findViewById(R.id.btnGuanGua)
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
        btnGuanGua.setOnClickListener { showDetail = !showDetail; updateUI() }
        btnReset.setOnClickListener { engine.reset(); running = true; showDetail = false; updateUI() }

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

        // 卦象 & 行为
        val hex = s.hexagram
        val guaName = Hexagram.NAMES[hex]
        val upTri = Hexagram.TRIGRAM_NAMES[Hexagram.upperTrigram(hex)]
        val loTri = Hexagram.TRIGRAM_NAMES[Hexagram.lowerTrigram(hex)]

        if (s.changingLine >= 0) {
            val lineIdx = s.changingLine
            val isYang = s.lineYang(lineIdx)
            val lineLabel = Hexagram.lineName(lineIdx, isYang)
            val targetName = Hexagram.NAMES[s.targetHexagram]
            tvBehavior.text = guaName + " → " + targetName + " (" + s.behaviorName() + ")"
        } else {
            tvBehavior.text = guaName + " · " + s.behaviorName()
        }

        tvAge.text = "年龄: " + s.age
        tvChildren.text = "子代: " + s.children + " (第" + s.generation + "代)"
        pbAtp.progress = (s.atp / 10f * 100).toInt()
        pbGlucose.progress = (s.glucose / 20f * 100).toInt()
        pbDamage.progress = (s.damage * 100).toInt()
        tvStatus.text = "ATP: " + "%.1f".format(s.atp) + "/10 | 葡萄糖: " + "%.1f".format(s.glucose) + "/20 | 损伤: " + "%.0f".format(s.damage * 100) + "%"

        // 卦象详情
        if (showDetail) {
            val judgement = Hexagram.JUDGMENTS[hex]
            val lines = StringBuilder()
            for (i in 0 until 6) {
                val ly = s.lineYang(i)
                val ln = Hexagram.lineName(i, ly)
                val w = "%.2f".format(engine.weights[hex][i])
                lines.append(ln + "(" + w + ") ")
            }
            tvHexagram.text = loTri + "下" + upTri + "上 | " + judgement +
                "\n变爻权重: " + lines.toString() +
                "\n语料: " + engine.experienceCount + " 条"
        } else {
            tvHexagram.text = loTri + "下" + upTri + "上 · " + guaName + "卦 · 语料" + engine.experienceCount + "条"
        }
    }

    override fun onDestroy() {
        running = false
        engine.close()
        super.onDestroy()
    }
}
