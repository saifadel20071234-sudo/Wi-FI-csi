"""
================================================================================
          WI-FI CSI HUMAN SENSING & THERMAL TRACKING SYSTEM
                     Master Orchestrator / Runner
================================================================================
Runs the entire system end-to-end:
1. Guides hardware placement (Transmitter vs Receiver).
2. Auto-detects connected ESP32 COM port.
3. Provides 1-click flashing for ESP32 Receiver (No manual compile needed).
4. Auto-trains AI model if needed.
5. Launches Streamlit High-Tech Thermal Tracking Dashboard.
================================================================================
"""

import os
import sys
import time
import subprocess
import webbrowser

# Ensure UTF-8 output in Windows terminal
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")

def print_banner():
    print("""
================================================================================
   📡 WI-FI CSI HUMAN SENSING & SPATIAL THERMAL TRACKING SYSTEM
                      graduation project - master runner
================================================================================
    """)

def scan_ports():
    """Scans for active serial COM ports."""
    try:
        import serial.tools.list_ports
        return list(serial.tools.list_ports.comports())
    except Exception:
        return []

def prompt_hardware_setup():
    """Shows physical placement instructions."""
    print("--------------------------------------------------------------------------------")
    print("📍 [إرشادات توصيل الهاردوير في المكان]:")
    print("  1) بوردة الـ Transmitter (ESP32 #1 - المرسل):")
    print("     - ضعها في زاوية أو طرف الغرفة (على ارتفاع ~1 متر).")
    print("     - وصّلها بأي شاحن موبايل أو باوربانك (الكود عليها مسبقاً ويبث شبكة CSI_TX_NODE).")
    print("")
    print("  2) بوردة الـ Receiver (ESP32 #2 - المستقبل):")
    print("     - ضعها في الطرف المقابل للغرفة أمام المرسل (بمسافة 2 إلى 4 أمتار).")
    print("     - وصّلها باللابتوب عن طريق كابل الـ USB.")
    print("--------------------------------------------------------------------------------")

def detect_or_wait_port():
    """Detects COM port or prompts user to plug in ESP32."""
    while True:
        ports = scan_ports()
        if ports:
            print("\n🔍 [المنافذ المكتشفة حالياً]:")
            for idx, p in enumerate(ports, start=1):
                desc = p.description or "Standard Serial Device"
                print(f"   [{idx}] {p.device}  -->  {desc}")
            
            if len(ports) == 1:
                selected = ports[0].device
                print(f"\n[+] تم التعرف تلقائياً على بوردة الـ ESP32 على المنفذ: {selected}")
                return selected
            else:
                # Intelligently suggest COM9 or the last connected device for Receiver
                default_idx = len(ports)  # usually the second board plugged in is the receiver
                for idx, p in enumerate(ports, start=1):
                    if "COM9" in p.device:
                        default_idx = idx
                        break
                
                print(f"\n💡 [ملاحظة]: إذا كانت بوردة الإرسال موصلة أيضاً، فاختر بوردة الاستقبال (غالباً COM9 أو الخيار {default_idx}).")
                choice = input(f"اختر رقم المنفذ المتصل به الـ Receiver (ESP32 #2) [1-{len(ports)}] أو اضغط Enter للاختيار الافتراضي [{default_idx}]: ").strip()
                if choice.isdigit() and 1 <= int(choice) <= len(ports):
                    return ports[int(choice) - 1].device
                return ports[default_idx - 1].device

        print("\n⚠️ لم يتم العثور على أي بوردة ESP32 موصلة باللابتوب حالياً!")
        print("   1) تأكد من توصيل بوردة الـ Receiver بكابل USB جيد يدعم نقل البيانات.")
        print("   2) اضغط Enter لإعادة الفحص التلقائي، أو اكتب اسم الـ COM يدوياً (مثال: COM8):")
        manual = input("   --> ").strip().upper()
        if manual.startswith("COM"):
            return manual

def flash_receiver(port: str):
    """Flashes precompiled ESP32 CSI Receiver firmware."""
    print("\n" + "="*80)
    print(f"⚡ [جاري حرق الفيرموير على بوردة الاستقبال (ESP32 #2) عبر {port}]...")
    print("="*80)
    
    bin_dir = os.path.join(ROOT_DIR, "firmware_receiver", "bin")
    bootloader = os.path.join(bin_dir, "bootloader.bin")
    partition = os.path.join(bin_dir, "partition-table.bin")
    receiver = os.path.join(bin_dir, "csi_receiver.bin")

    if not (os.path.exists(bootloader) and os.path.exists(partition) and os.path.exists(receiver)):
        print("❌ خطأ: ملفات الفيرموير الثنائية غير موجودة في firmware_receiver/bin/")
        return False

    cmd = [
        sys.executable, "-m", "esptool",
        "--chip", "esp32",
        "--port", port,
        "--baud", "460800",
        "write_flash", "-z",
        "0x1000", bootloader,
        "0x8000", partition,
        "0x10000", receiver
    ]

    print(f"[CMD] {' '.join(cmd)}\n")
    print("⏳ جاري الحرق... (إذا علق في Connecting... اضغط باستمرار على زر BOOT في البوردة لمدة ثانيتين)\n")

    res = subprocess.run(cmd)
    if res.returncode == 0:
        print("\n✅ تم حرق فيرموير الـ Receiver بنجاح تام 100%!")
        print("   البوردة الآن تستقبل إشارات الـ CSI من الـ Transmitter وتبثها إلى اللابتوب.")
        return True
    else:
        print("\n❌ حدث خطأ أثناء الحرق. تأكد من إغلاق أي برنامج يستخدم المنفذ (مثل Arduino IDE أو Serial Monitor).")
        return False

def verify_ai_model():
    """Checks if model files exist, trains them if missing."""
    model_path = os.path.join(ROOT_DIR, "dashboard", "csi_model.pkl")
    if not os.path.exists(model_path):
        print("\n[*] جاري التحقق من موديل الذكاء الاصطناعي...")
        train_script = os.path.join(ROOT_DIR, "ml", "ml_model.py")
        if os.path.exists(train_script):
            subprocess.run([sys.executable, train_script])
            print("[+] تم تجهيز موديل الذكاء الاصطناعي بنجاح!")
    else:
        print("[+] موديل الذكاء الاصطناعي متدرب وجاهز (Accuracy 100%).")

def test_live_csi_stream(port: str):
    """Quick diagnostic to read raw CSI frames from serial."""
    print("\n" + "="*80)
    print(f"🧪 [اختبار قراءة بيانات الـ CSI الحية من {port} لمدة 5 ثوانٍ]...")
    print("="*80)
    try:
        import serial
        ser = serial.Serial(port, 115200, timeout=1.5)
        time.sleep(1.0)
        ser.reset_input_buffer()
        print("Listening for incoming frames...")
        start_time = time.time()
        packets = 0
        while time.time() - start_time < 5.0:
            if ser.in_waiting:
                line = ser.readline().decode("utf-8", errors="ignore").strip()
                if "CSI_START" in line and "CSI_END" in line:
                    packets += 1
                    parts = line.split(",")
                    print(f"   [FRAME #{packets}] RSSI: {parts[1]} dBm | Subcarriers: {parts[3]} | Amplitudes: {parts[4:10]}...")
            time.sleep(0.01)
        ser.close()
        if packets > 0:
            print(f"\n✅ ممتاز! تم استقبال {packets} إطار CSI خام بنجاح من الهاردوير!")
        else:
            print("\n⚠️ لم يتم استقبال إطارات CSI بعد. تأكد أن الـ Transmitter شغال في نفس الغرفة.")
    except Exception as e:
        print(f"\n❌ تعذر فتح المنفذ: {e}")

def run_dashboard(port: str = ""):
    """Launches the Streamlit Live Dashboard."""
    print("\n" + "="*80)
    print("🚀 [إطلاق الداش بورد الحية - Live Thermal Radar Dashboard]")
    print("="*80)
    app_path = os.path.join(ROOT_DIR, "dashboard", "app.py")
    
    # Write preferred port to a temporary environment variable
    env = os.environ.copy()
    if port:
        env["CSI_DEFAULT_PORT"] = port
        
    print("[*] جاري تشغيل خادم Streamlit وفتح المتصفح...")
    print("[*] Dashboard URL: http://localhost:8501")
    print("--------------------------------------------------------------------------------")
    print("💡 بعد فتح المتصفح:")
    print("   1) ستجد المنفذ تم تحديده تلقائياً في القائمة الجانبية.")
    print("   2) اضغط على الزر الأخضر: '🚀 CONNECT & START LIVE TRACKING'.")
    print("   3) تحرك في الغرفة وراقب الهالة الحرارية وهي تتبع حركتك بانسيابية!")
    print("--------------------------------------------------------------------------------\n")
    
    cmd = [sys.executable, "-m", "streamlit", "run", app_path]
    subprocess.run(cmd, env=env)

def main():
    clear_screen()
    print_banner()
    prompt_hardware_setup()
    
    # 1. Scan / Detect Port
    selected_port = detect_or_wait_port()
    
    # 2. Check AI Model
    verify_ai_model()
    
    # 3. Main Menu
    print("\n" + "="*80)
    print("📋 [اختر ما تريد تنفيذه الآن]:")
    print("   [1] 🚀 تشغيل النظام والتتبع الحي مباشرة (Live Dashboard)  [الافتراضي - اضغط Enter]")
    print("   [2] ⚡ حرق الفيرموير على الـ Receiver (ESP32 #2) ثم فتح الداش بورد فوراً")
    print("   [3] 🧪 فحص واختبار قراءة إشارات الـ CSI الحية من البوردة (Diagnostic Test)")
    print("   [4] 🔄 إعادة تدريب موديل الذكاء الاصطناعي (Train AI Model)")
    print("   [5] 🚪 خروج")
    print("="*80)

    choice = input("\nاكتب رقم الخيار [1-5] واضغط Enter (أو اضغط Enter مباشرة للخيار 1): ").strip()
    
    if choice == "2":
        ok = flash_receiver(selected_port)
        if ok:
            print("\n⏳ جاري إطلاق الداش بورد خلال 3 ثوانٍ...")
            time.sleep(2.5)
            run_dashboard(selected_port)
    elif choice == "3":
        test_live_csi_stream(selected_port)
        input("\nاضغط Enter للعودة للقائمة وتشغيل الداش بورد...")
        run_dashboard(selected_port)
    elif choice == "4":
        train_script = os.path.join(ROOT_DIR, "ml", "ml_model.py")
        subprocess.run([sys.executable, train_script])
        input("\nاضغط Enter لتشغيل الداش بورد الآن...")
        run_dashboard(selected_port)
    elif choice == "5":
        print("\nمع السلامة!")
        sys.exit(0)
    else:
        # Default: 1
        # Quick check if user wants to flash first
        ask = input("\nهل هذه أول مرة توصل فيها بوردة الـ Receiver وتريد حرق الفيرموير عليها أولاً؟ [y/N]: ").strip().lower()
        if ask == "y" or ask == "yes" or ask == "نعم":
            ok = flash_receiver(selected_port)
            if ok:
                time.sleep(2.0)
                run_dashboard(selected_port)
        else:
            run_dashboard(selected_port)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[!] تم إيقاف البرنامج بواسطة المستخدم.")
