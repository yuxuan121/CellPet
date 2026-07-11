package com.operit.cellpet

import android.app.*
import android.content.Intent
import android.os.IBinder
import androidx.core.app.NotificationCompat
import java.util.*

class CellService : Service() {
    private lateinit var engine: CellEngine
    private val timer = Timer()

    override fun onCreate() {
        super.onCreate()
        try { engine = CellEngine.getInstance(this) } catch (e: Throwable) {}
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        try {
            val channel = NotificationChannel("cell_life", "细胞生命", NotificationManager.IMPORTANCE_LOW)
            getSystemService(NotificationManager::class.java).createNotificationChannel(channel)
            val notif = NotificationCompat.Builder(this, "cell_life")
                .setContentTitle("细胞宠物")
                .setContentText("细胞存活中…")
                .setSmallIcon(android.R.drawable.ic_menu_manage)
                .setOngoing(true)
                .build()
            // No startForeground - just regular service for now
            timer.scheduleAtFixedRate(object : TimerTask() {
                override fun run() {
                    try {
                        val s = engine.tick()
                        val nm = getSystemService(NotificationManager::class.java) as NotificationManager
                        if (s.alive) {
                            val n = NotificationCompat.Builder(this@CellService, "cell_life")
                                .setContentTitle(s.behaviorName() + " | ATP: " + "%.1f".format(s.atp))
                                .setContentText("子代: " + s.children + " | 损伤: " + "%.0f".format(s.damage * 100) + "%")
                                .setSmallIcon(android.R.drawable.ic_menu_manage)
                                .setOngoing(true)
                                .build()
                            nm.notify(1, n)
                        } else {
                            val n = NotificationCompat.Builder(this@CellService, "cell_life")
                                .setContentTitle("细胞凋亡")
                                .setContentText("存活" + s.age + "轮，子代" + s.children + "个")
                                .setSmallIcon(android.R.drawable.ic_menu_manage)
                                .build()
                            nm.notify(1, n)
                            stopSelf()
                        }
                    } catch (e: Throwable) {}
                }
            }, 0, 5000)
        } catch (e: Throwable) {}
        return START_STICKY
    }

    override fun onBind(intent: Intent?): IBinder? = null
    override fun onDestroy() { timer.cancel(); engine.close(); super.onDestroy() }
}