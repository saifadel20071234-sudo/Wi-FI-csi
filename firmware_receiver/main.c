#include <stdio.h>
#include <math.h>
#include <stdint.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "esp_err.h"
#include "esp_event.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_wifi.h"
#include "nvs_flash.h"

#define TAG      "CSI_RX"
#define TX_SSID  "CSI_TX_NODE"
#define TX_PASS  "12345678"

static void csi_callback(void *ctx, wifi_csi_info_t *data)
{
    if (data == NULL || data->buf == NULL || data->len < 2) {
        return;
    }

    const int8_t *buf = data->buf;
    const int len = data->len;
    const int rssi = data->rx_ctrl.rssi;
    const uint32_t timestamp_ms =
        (uint32_t)(xTaskGetTickCount() * portTICK_PERIOD_MS);
    const int num_sub = len / 2;
    const int output_sub = (num_sub >= 64) ? 64 : num_sub;
    if (output_sub < 10) return;

    // Python script accepts packets containing exactly 64 amplitudes.
    printf("CSI_START,%lu,%d,%d",
           (unsigned long)timestamp_ms, rssi, output_sub);

    for (int i = 0; i < output_sub; i++) {
        int idx = (i < num_sub) ? (i * 2) : ((i % num_sub) * 2);
        const float imag = (float)buf[idx];
        const float real = (float)buf[idx + 1];
        const float amp = sqrtf(real * real + imag * imag);
        printf(",%.2f", amp);
    }

    printf(",CSI_END\n");
    fflush(stdout);
}

static void wifi_event_handler(void *arg,
                               esp_event_base_t event_base,
                               int32_t event_id,
                               void *event_data)
{
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
        ESP_LOGI(TAG, "Connecting to transmitter SSID: %s", TX_SSID);
        ESP_ERROR_CHECK(esp_wifi_connect());
    } else if (event_base == WIFI_EVENT &&
               event_id == WIFI_EVENT_STA_DISCONNECTED) {
        ESP_LOGW(TAG, "Disconnected; retrying connection");
        esp_wifi_connect();
    } else if (event_base == IP_EVENT &&
               event_id == IP_EVENT_STA_GOT_IP) {
        ESP_LOGI(TAG, "Connected. Waiting for CSI packets...");
    }
}

static void wifi_init_sta(void)
{
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_t *sta_netif = esp_netif_create_default_wifi_sta();
    if (!sta_netif) {
        ESP_LOGE(TAG, "Failed to create WiFi STA netif");
        return;
    }

    wifi_init_config_t init_config = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&init_config));

    ESP_ERROR_CHECK(esp_event_handler_register(
        WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_event_handler, NULL));
    ESP_ERROR_CHECK(esp_event_handler_register(
        IP_EVENT, IP_EVENT_STA_GOT_IP, &wifi_event_handler, NULL));

    wifi_csi_config_t csi_config = {
        .lltf_en = true,
        .htltf_en = true,
        .stbc_htltf2_en = true,
        .ltf_merge_en = true,
        .channel_filter_en = false,
        .manu_scale = false,
    };

    wifi_config_t station_config = {
        .sta = {
            .ssid = TX_SSID,
            .password = TX_PASS,
        },
    };

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &station_config));
    ESP_ERROR_CHECK(esp_wifi_set_csi_config(&csi_config));
    ESP_ERROR_CHECK(esp_wifi_set_csi_rx_cb(&csi_callback, NULL));
    ESP_ERROR_CHECK(esp_wifi_set_csi(true));
    ESP_ERROR_CHECK(esp_wifi_start());
}

void app_main(void)
{
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES ||
        ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    wifi_init_sta();
    ESP_LOGI(TAG, "CSI receiver initialized");
}
