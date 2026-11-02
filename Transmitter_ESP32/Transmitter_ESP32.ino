

#include <WiFi.h>
#include <WiFiUdp.h>


const char* ssid     = "CSI_TX_NODE";
const char* password = "12345678";
const int   channel  = 6;

WiFiUDP udp;
const int udpPort = 1234;
IPAddress broadcastIp(255, 255, 255, 255);

unsigned long packetCount = 0;

void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println("\n==========================================");
  Serial.println("  📡 Wi-Fi CSI Transmitter Node Started!");
  Serial.println("  👤 Member 1 Task - Active");
  Serial.println("==========================================");

  // إعداد نقطة الواي فاي (Access Point)
  WiFi.mode(WIFI_AP);
  WiFi.softAP(ssid, password, channel);

  // ضبط قوة الإرسال إلى أقصى حد (19.5 - 20 dBm)
  WiFi.setTxPower(WIFI_POWER_19_5dBm);

  // تشغيل بروتوكول UDP
  udp.begin(udpPort);

  Serial.print("✅ SSID Network Name: ");
  Serial.println(ssid);
  Serial.print("✅ IP Address: ");
  Serial.println(WiFi.softAPIP());
  Serial.print("✅ Channel: ");
  Serial.println(channel);
  Serial.println("🚀 Transmitting CSI Beacon Packets...\n");
}

void loop() {
  // إرسال حزمة بيانات مستمرة كل 50 ملي ثانية (20 حزمة في الثانية)
  udp.beginPacket(broadcastIp, udpPort);
  udp.printf("PING:%lu", packetCount++);
  udp.endPacket();

  // طباعة تأكيد كل 50 حزمة على الـ Serial Monitor
  if (packetCount % 50 == 0) {
    Serial.printf("📡 Status: Transmitting normally... Total Packets: %lu\n", packetCount);
  }

  delay(50); // تأخير 50ms = 20 packets/sec
}
