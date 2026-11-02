#include <WiFi.h>
#include <WiFiUdp.h>
#include "esp_wifi.h"
#include "ping/ping_sock.h"
#include <math.h>

const char* ssid     = "CSI_TX_NODE";
const char* password = "12345678";
const int   channel  = 6;
#define TARGET_SUBCARRIERS 64
#define LED_PIN 2

static char csi_line[700];
static esp_ping_handle_t ping_handle = NULL;

void _csi_rx_callback(void *ctx, wifi_csi_info_t *data) {
    if (data == NULL || data->buf == NULL || data->len < 2) return;

    // Filter packets: accept if from our transmitter MAC (ends with 4b:59:1d)
    if (data->mac[3] != 0x4b || data->mac[4] != 0x59 || data->mac[5] != 0x1d) {
        return;
    }

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

void startContinuousPing() {
    if (ping_handle != NULL) return;

    esp_ping_config_t ping_config = ESP_PING_DEFAULT_CONFIG();
    ip_addr_t target_addr;
    ipaddr_aton("192.168.4.1", &target_addr);
    ping_config.target_addr = target_addr;
    ping_config.interval_ms = 50; // 20 pings per second
    ping_config.count = 0;       // Continuous forever

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
}

void setup() {
    pinMode(LED_PIN, OUTPUT);
    digitalWrite(LED_PIN, LOW);

    Serial.setTxBufferSize(2048);
    Serial.begin(115200);
    delay(400);

    Serial.println("\n==========================================");
    Serial.println("  Wi-Fi CSI Receiver Active (Active Ping Engine)");
    Serial.println("==========================================");

    WiFi.onEvent([](WiFiEvent_t event, WiFiEventInfo_t info) {
        if (event == ARDUINO_EVENT_WIFI_STA_GOT_IP) {
            digitalWrite(LED_PIN, HIGH);
            setupCSI();
            startContinuousPing();
            Serial.print("Connected! IP: ");
            Serial.println(WiFi.localIP());
            Serial.println("CSI Engine Initialized & Ping Active!");
        } else if (event == ARDUINO_EVENT_WIFI_STA_DISCONNECTED) {
            digitalWrite(LED_PIN, LOW);
            Serial.print("[WiFi] Disconnected, reason: ");
            Serial.println(info.wifi_sta_disconnected.reason);
            WiFi.begin(ssid, password, channel);
        }
    });

    WiFi.mode(WIFI_STA);
    WiFi.setAutoReconnect(true);
    WiFi.setSleep(false);
    WiFi.begin(ssid, password, channel);
}

void loop() {
    if (WiFi.status() == WL_CONNECTED) {
        digitalWrite(LED_PIN, HIGH);
    } else {
        digitalWrite(LED_PIN, LOW);
    }
    delay(100);
}
