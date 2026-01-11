# lvgl_demo

## Overview
This project is a demonstration of the LVGL library, a lightweight graphics library for embedded systems. It provides a simple and efficient way to create graphical user interfaces (GUIs) for embedded systems. The library is designed to be easy to use and highly efficient, making it ideal for use in resource-constrained environments.

## Supported Platforms and Interfaces
- [ ] T3: SPI
- [x] T5AI: RGB/8080/SPI/QSPI
- [ ] ESP32

## Supported Drivers

### Screen
- SPI
- [x] ST7789
- [x] ILI9341
- [x] GC9A01

- RGB
- [x] ILI9488

- QSPI
- [x] CO5300

### Touch

- I2C
- [x] GT911
- [x] CST816
- [x] GT1511
- [x] CST9217

### Rotary Encoder

## Supported Development Board List

| Development Board | Screen Interface and Driver | Touch Interface and Driver | Touch Pins | Remarks |

| -------- | -------- | -------- | -------- | -------- | | T5-E1-Touch-AMOLED-1.75 | RGB565/CO5300 | I2C/CST9217 | SCL(P20)/SDA(P21) | [https://developer.tuya.com/cn/docs/iot-device-dev/T5-E1-IPEX-development-board?id=Ke9xehig1cabj](https://developer.tuya.com/cn/docs/iot-device-dev/T5-E1-IPEX-development-board?id=Ke9xehig1cabj) |

> More driver adaptations and testing are underway...

## Usage Flow

1. Run `tos menuconfig` to configure the project

2. Configure the corresponding screen/touch/RTC/QMI/SD CARD/WIFI/GPS/CODEC/GPIO drivers, etc.

3. Configure the corresponding GPIO pins

4. Compile the project: `tos.py build`

5. Flash and run: `tos.py flash`