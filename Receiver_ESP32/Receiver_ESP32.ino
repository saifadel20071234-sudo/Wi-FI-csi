#include <WiFi.h>
#include "esp_wifi.h"
#include <math.h>

const char* ssid     = "CSI_TX_NODE";
const char* password = "12345678";

#define NUM_SUBCARRIERS 64

// CSI Callback executed on core 0 by the Wi-Fi task
void _csi_rx_callback(void *ctx, wifi_csi_info_t *data) {
    if (data == NULL || data->buf == NULL || data->len < 2) {
        return;
    }

    const int8_t *buf = data->buf;
    const int len = data->len;
    const int rssi = data->rx_ctrl.rssi;
    const uint32_t timestamp_ms = millis();
    const int num_sub = len / 2;

    // We output CSI frames only when they match 64 subcarriers
    if (num_sub != NUM_SUBCARRIERS) {
        return;
    }

    Serial.printf("CSI_START,%lu,%d,%d", (unsigned long)timestamp_ms, rssi, num_sub);

    for (int i = 0; i + 1 < len; i += 2) {
        float imag = (float)buf[i];
        float real = (float)buf[i + 1];
        float amp = sqrtf(real * real + imag * imag);
        Serial.printf(",%.2f", amp);
    }

    Serial.println(",CSI_END");
}

void setup() {
    Serial.begin(115200);
    delay(1000);

    Serial.println("\n==========================================");
    Serial.println("  📥 Wi-Fi CSI Receiver Starting...");
    Serial.println("==========================================");

    // 1. Initialize Wi-Fi in Station Mode
    WiFi.mode(WIFI_STA);
    WiFi.disconnect();
    delay(100);

    // 2. Connect to Transmitter AP
    Serial.print("Connecting to Transmitter SSID: ");
    Serial.println(ssid);
    WiFi.begin(ssid, password);

    // 3. Configure CSI (MUST be done after Wi-Fi is initialized/started)
    wifi_csi_config_t csi_config = {
        .lltf_en = true,
        .htltf_en = true,
        .stbc_htltf2_en = true,
        .ltf_merge_en = true,
        .channel_filter_en = false,
        .manu_scale = false,
    };

    esp_err_t err;
    err = esp_wifi_set_csi_config(&csi_config);
    if (err != ESP_OK) {
        Serial.printf("[-] esp_wifi_set_csi_config failed: 0x%x\n", err);
    }

    err = esp_wifi_set_csi_rx_cb(&_csi_rx_callback, NULL);
    if (err != ESP_OK) {
        Serial.printf("[-] esp_wifi_set_csi_rx_cb failed: 0x%x\n", err);
    }

    err = esp_wifi_set_csi(true);
    if (err != ESP_OK) {
        Serial.printf("[-] esp_wifi_set_csi failed: 0x%x\n", err);
    } else {
        Serial.println("✅ CSI Engine Activated Successfully!");
    }

    Serial.println("Waiting for Wi-Fi connection and CSI packets...");
}

void loop() {
    static unsigned long last_check = 0;
    if (millis() - last_check > 3000) {
        last_check = millis();
        if (WiFi.status() == WL_CONNECTED) {
            // Quietly connected
        } else {
            Serial.println("[*] Reconnecting to CSI_TX_NODE...");
            WiFi.reconnect();
        }
    }
    delay(10);
}
