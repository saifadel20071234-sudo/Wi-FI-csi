# Keep the source at firmware_receiver/main.c so existing flashing scripts and
# documentation retain their paths. ESP-IDF accepts this relative source path.
idf_component_register(SRCS "../main.c" INCLUDE_DIRS ".")
