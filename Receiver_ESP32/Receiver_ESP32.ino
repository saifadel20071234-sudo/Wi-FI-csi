#include <WiFi.h>
#include <WiFiUdp.h>
#include "esp_wifi.h"
#include "ping/ping_sock.h"
#include <math.h>

#define TARGET_SUBCARRIERS 64
#define LED_PIN 2

const char* ssid_tx  = "CSI_TX_NODE";
const char* pass_tx  = "12345678";
const char* ssid_ext = "CSI_NODE_EXT";
const char* pass_ext = "12345678";
const int   channel  = 6;

static char csi_line[700];
static esp_ping_handle_t ping_handle = NULL;
static bool csi_initialized = false;
static unsigned long last_connect_attempt = 0;
static unsigned long last_udp_probe = 0;
static int current_network_idx = 0;
static WiFiUDP udpProbe;

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
    delay(300);

    Serial.println("\n========================================================");
    Serial.println("  Wi-Fi CSI Fast Multi-Node Receiver (Zero-Delay)");
    Serial.println("  Nodes: TX (CSI_TX_NODE) + EXT (CSI_NODE_EXT) + RX (COM)");
    Serial.println("========================================================");

    WiFi.onEvent([](WiFiEvent_t event, WiFiEventInfo_t info) {
        if (event == ARDUINO_EVENT_WIFI_STA_GOT_IP) {
            digitalWrite(LED_PIN, HIGH);
            setupCSI();
            startContinuousPing(WiFi.gatewayIP());
            Serial.print("\n[WiFi] Connected to: ");
            Serial.print(WiFi.SSID());
            Serial.print(" | IP: ");
            Serial.print(WiFi.localIP());
            Serial.print(" | Gateway: ");
            Serial.println(WiFi.gatewayIP());
            Serial.println("CSI Engine Active & Multi-Node Broadcast Probing Started!");
        } else if (event == ARDUINO_EVENT_WIFI_STA_DISCONNECTED) {
            digitalWrite(LED_PIN, LOW);
            csi_initialized = false;
        }
    });

    WiFi.mode(WIFI_STA);
    WiFi.setAutoReconnect(true);
    WiFi.setSleep(false);

    Serial.println("[WiFi] Fast-connecting directly on Channel 6...");
    WiFi.begin(ssid_tx, pass_tx, channel);
    last_connect_attempt = millis();
}

void loop() {
    if (WiFi.status() == WL_CONNECTED) {
        digitalWrite(LED_PIN, HIGH);

        // Active multi-node broadcast probing: stimulates TX and EXT AP simultaneously
        if (millis() - last_udp_probe >= 40) {
            last_udp_probe = millis();
            udpProbe.beginPacket(IPAddress(255, 255, 255, 255), 12345);
            udpProbe.write((const uint8_t*)"CSI_PROBE", 9);
            udpProbe.endPacket();
        }
    } else {
        digitalWrite(LED_PIN, LOW);
        if (millis() - last_connect_attempt > 3500) {
            current_network_idx = (current_network_idx + 1) % 2;
            const char* next_ssid = (current_network_idx == 0) ? ssid_tx : ssid_ext;
            const char* next_pass = (current_network_idx == 0) ? pass_tx : pass_ext;
            Serial.print("[WiFi] Retrying with: ");
            Serial.println(next_ssid);
            WiFi.disconnect();
            WiFi.begin(next_ssid, next_pass, channel);
            last_connect_attempt = millis();
        }
    }
    delay(10);
}
