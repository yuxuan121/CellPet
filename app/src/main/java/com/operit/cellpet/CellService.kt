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
        engine = CellEngine.getInstance(this)
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val channel = NotificationChannel("cell_life", "Cell Life", NotificationManager.IMPORTANCE_LOW)
        getSystemService(NotificationManager::class.java).createNotificationChannel(channel)
        val notif = NotificationCompat.Builder(this, "cell_life")
            .setContentTitle("Cell Pet")
            .setContentText("Cell is alive...")
            .setSmallIcon(android.R.drawable.ic_menu_manage)
            .setOngoing(true)
            .build()
        startForeground(1, notif)

        timer.scheduleAtFixedRate(object : TimerTask() {
            override fun run() {
                val s = engine.tick()
                val nm = getSystemService(NotificationManager::class.java) as NotificationManager
                if (s.alive) {
                    val n = NotificationCompat.Builder(this@CellService, "cell_life")
                        .setContentTitle(s.behaviorName() + " | ATP: " + "%.1f".format(s.atp))
                        .setContentText("Children: " + s.children + " | Damage: " + "%.0f".format(s.damage * 100) + "%")
                        .setSmallIcon(android.R.drawable.ic_menu_manage)
                        .setOngoing(true)
                        .build()
                    nm.notify(1, n)
                } else {
                    val n = NotificationCompat.Builder(this@CellService, "cell_life")
                        .setContentTitle("Cell Died")
                        .setContentText("Lived " + s.age + " ticks, had " + s.children + " children")
                        .setSmallIcon(android.R.drawable.ic_menu_manage)
                        .build()
                    nm.notify(1, n)
                    stopSelf()
                }
            }
        }, 0, 5000)
        return START_STICKY
    }

    override fun onBind(intent: Intent?): IBinder? = null
    override fun onDestroy() { timer.cancel(); engine.close(); super.onDestroy() }
}