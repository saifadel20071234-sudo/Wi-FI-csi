#include <WiFi.h>
#include <WiFiMulti.h>
#include "esp_wifi.h"
#include "ping/ping_sock.h"
#include <math.h>

#define TARGET_SUBCARRIERS 64
#define LED_PIN 2

static WiFiMulti wifiMulti;
static char csi_line[700];
static esp_ping_handle_t ping_handle = NULL;
static bool csi_initialized = false;

void _csi_rx_callback(void *ctx, wifi_csi_info_t *data) {
    if (data == NULL || data->buf == NULL || data->len < 2) return;

    int len = data->len;
    int raw_sub = len / 2;
    if (raw_sub <= 0) return;

    int8_t *buf = data->buf;
    int rssi = data->rx_ctrl.rssi;
    uint32_t ts = millis();

    int n = snprintf(csi_line, sizeof(csi_line), "CSI_START,%lu,%d,%d", (unsigned long)ts, rssi, TARGET_SUBCARRIERS);

    for (int i = 0; i < TARGET_SUBCARRIERS; i++) {
        int idx = (i < raw_sub) ? (i * 2) : ((i % raw_sub) * 2);
        float imag = (float)buf[idx];
        float real = (float)buf[idx + 1];
        float amp = sqrtf(real * real + imag * imag);
        n += snprintf(csi_line + n, sizeof(csi_line) - n, ",%.1f", amp);
    }
    snprintf(csi_line + n, sizeof(csi_line) - n, ",CSI_END\n");
    Serial.print(csi_line);
}

void startContinuousPing(IPAddress gateway) {
    if (ping_handle != NULL) {
        esp_ping_stop(ping_handle);
        esp_ping_delete_session(ping_handle);
        ping_handle = NULL;
    }

    esp_ping_config_t ping_config = ESP_PING_DEFAULT_CONFIG();
    ip_addr_t target_addr;
    if (gateway[0] == 0) {
        ipaddr_aton("192.168.4.1", &target_addr);
    } else {
        ipaddr_aton(gateway.toString().c_str(), &target_addr);
    }
    ping_config.target_addr = target_addr;
    ping_config.interval_ms = 50; // 20 pings per second
    ping_config.count = 0;        // Continuous forever

    esp_ping_callbacks_t cbs = {
        .cb_args = NULL,
        .on_ping_success = NULL,
        .on_ping_timeout = NULL,
        .on_ping_end = NULL
    };

    esp_ping_new_session(&ping_config, &cbs, &ping_handle);
    esp_ping_start(ping_handle);
}

void setupCSI() {
    if (csi_initialized) return;
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
    csi_initialized = true;
}

void setup() {
    pinMode(LED_PIN, OUTPUT);
    digitalWrite(LED_PIN, LOW);

    Serial.setTxBufferSize(2048);
    Serial.begin(115200);
    delay(400);

    Serial.println("\n========================================================");
    Serial.println("  Wi-Fi CSI Multi-Node Receiver (Active Dual AP Engine)");
    Serial.println("  Nodes: TX (CSI_TX_NODE) + EXT (CSI_NODE_EXT) + RX (COM)");
    Serial.println("========================================================");

    WiFi.mode(WIFI_STA);
    WiFi.setSleep(false);

    // Register primary Transmitter AP and Extender AP
    wifiMulti.addAP("CSI_TX_NODE", "12345678");
    wifiMulti.addAP("CSI_NODE_EXT", "12345678");
    wifiMulti.addAP("CSI_NODE_EXT", ""); // In case open network
}

void loop() {
    if (wifiMulti.run() == WL_CONNECTED) {
        digitalWrite(LED_PIN, HIGH);
        if (!csi_initialized) {
            setupCSI();
            startContinuousPing(WiFi.gatewayIP());
            Serial.print("[WiFi] Connected to: ");
            Serial.print(WiFi.SSID());
            Serial.print(" | IP: ");
            Serial.print(WiFi.localIP());
            Serial.print(" | Gateway: ");
            Serial.println(WiFi.gatewayIP());
            Serial.println("CSI Engine Active & 20Hz Unicast Ping Streaming!");
        }
    } else {
        digitalWrite(LED_PIN, LOW);
        csi_initialized = false;
        delay(200);
    }
    delay(50);
}
