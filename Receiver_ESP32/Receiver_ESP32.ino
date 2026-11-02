#include <WiFi.h>
#include "esp_wifi.h"
#include <math.h>

const char* ssid     = "CSI_TX_NODE";
const char* password = "12345678";
#define NUM_SUBCARRIERS 64

static char csi_line[600];

void _csi_rx_callback(void *ctx, wifi_csi_info_t *data) {
    if (data == NULL || data->buf == NULL || data->len < 2) return;
    int len = data->len;
    int num_sub = len / 2;
    if (num_sub != NUM_SUBCARRIERS) return;

    int8_t *buf = data->buf;
    int rssi = data->rx_ctrl.rssi;
    uint32_t ts = millis();

    int n = snprintf(csi_line, sizeof(csi_line), "CSI_START,%lu,%d,64", (unsigned long)ts, rssi);
    for (int i = 0; i + 1 < len; i += 2) {
        float imag = (float)buf[i];
        float real = (float)buf[i + 1];
        float amp = sqrtf(real * real + imag * imag);
        n += snprintf(csi_line + n, sizeof(csi_line) - n, ",%.1f", amp);
    }
    snprintf(csi_line + n, sizeof(csi_line) - n, ",CSI_END\n");
    Serial.print(csi_line);
}

void setup() {
    Serial.setTxBufferSize(2048);
    Serial.begin(115200);
    delay(500);

    Serial.println("\n==========================================");
    Serial.println("  📥 Wi-Fi CSI Receiver Starting...");
    Serial.println("==========================================");

    WiFi.mode(WIFI_STA);
    WiFi.setSleep(false);
    WiFi.begin(ssid, password);

    Serial.print("Connecting to: ");
    Serial.println(ssid);

    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 40) {
        delay(300);
        Serial.print(".");
        attempts++;
    }

    if (WiFi.status() == WL_CONNECTED) {
        Serial.println("\n✅ Connected to Transmitter! IP: " + WiFi.localIP().toString());
    } else {
        Serial.println("\n[!] Connection timeout, activating CSI anyway...");
    }

    // Enable CSI
    wifi_csi_config_t csi_config = {
        .lltf_en = true,
        .htltf_en = true,
        .stbc_htltf2_en = true,
        .ltf_merge_en = true,
        .channel_filter_en = false,
        .manu_scale = false,
    };

    esp_wifi_set_csi_config(&csi_config);
    esp_wifi_set_csi_rx_cb(&_csi_rx_callback, NULL);
    esp_wifi_set_csi(true);

    Serial.println("✅ CSI Engine Initialized & Streaming!");
}

void loop() {
    delay(100);
}
