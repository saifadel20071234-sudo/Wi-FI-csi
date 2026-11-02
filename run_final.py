"""
================================================================================
   📡 WI-FI CSI 3-NODE BIOMETRIC TRACKING SYSTEM - MASTER ORCHESTRATOR
================================================================================
Runs the entire system end-to-end with 1-Click Launch:
1. Guides 3-Node hardware placement (ESP32 TX + CSI_NODE_EXT + ESP32 RX).
2. Auto-detects connected ESP32 COM port (e.g. COM8).
3. Verifies AI Model & Calibrated Spatial Centroids.
4. Automatically launches Streamlit Live Biometric Thermal Radar.
5. Provides optional menu for firmware flashing or diagnostic testing.
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
   📡 WI-FI CSI 3-NODE BIOMETRIC HUMAN SENSING & THERMAL RADAR SYSTEM
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
    """Shows physical placement instructions for the 3-Node network."""
    print("--------------------------------------------------------------------------------")
    print("📍 [توزيع شبكة الاستشعار الثلاثية في الغرفة - 3-Node Network]:")
    print("  1) بوردة الإرسال (ESP32 #1 - TX):")
    print("     - ضعها في الزاوية اليسرى من الغرفة (تبث شبكة CSI_TX_NODE على القناة 6).")
    print("  2) جهاز تقوية وتوسيع الشبكة (CSI_NODE_EXT - Range Extender):")
    print("     - ضعه في الزاوية اليمنى من الغرفة لإنشاء تغطية هندسية كاملة.")
    print("  3) بوردة الاستقبال (ESP32 #2 - RX):")
    print("     - ضعها في منتصف الجدار المقابل (أمام المرسل والموسع) وموصولة باللابتوب.")
    print("--------------------------------------------------------------------------------")

def detect_or_wait_port():
    """Detects COM port or prompts user to plug in ESP32."""
    while True:
        ports = scan_ports()
        if ports:
            print("\n🔍 [المنافذ المكتشفة حالياً في اللابتوب]:")
            for idx, p in enumerate(ports, start=1):
                desc = p.description or "Standard Serial Device"
                print(f"   [{idx}] {p.device}  -->  {desc}")
            
            # Prefer COM8 or COM9 if present
            preferred = None
            for p in ports:
                if p.device in ("COM8", "COM9"):
                    preferred = p.device
                    break
                    
            if preferred:
                print(f"\n[+] تم التعرف التلقائي على بوردة الاستقبال على المنفذ: {preferred}")
                return preferred
            elif len(ports) == 1:
                selected = ports[0].device
                print(f"\n[+] تم العثور على بوردة الاستقبال على المنفذ: {selected}")
                return selected
            else:
                default_port = ports[-1].device
                print(f"\n💡 تم اختيار المنفذ الافتراضي: {default_port}")
                return default_port

        print("\n⚠️ لم يتم العثور على أي بوردة ESP32 موصلة باللابتوب حالياً!")
        print("   1) تأكد من توصيل بوردة الـ Receiver بكابل USB يدعم نقل البيانات.")
        print("   2) اضغط Enter لإعادة الفحص التلقائي، أو اكتب اسم الـ COM يدوياً (مثال: COM8):")
        manual = input("   --> ").strip().upper()
        if manual.startswith("COM"):
            return manual

def flash_receiver(port: str):
    """Flashes precompiled ESP32 CSI Receiver firmware."""
    print("\n" + "="*80)
    print(f"⚡ [جاري حرق الفيرموير المحدث على بوردة الاستقبال (ESP32 #2) عبر {port}]...")
    print("="*80)
    
    merged_bin = os.path.join(ROOT_DIR, "firmware_receiver", "bin", "Receiver_ESP32.ino.merged.bin")
    if not os.path.exists(merged_bin):
        merged_bin = os.path.join(ROOT_DIR, "Receiver_ESP32", "build", "esp32.esp32.esp32doit-devkit-v1", "Receiver_ESP32.ino.merged.bin")
    
    if os.path.exists(merged_bin):
        cmd = [
            sys.executable, "-m", "esptool",
            "--chip", "esp32",
            "--port", port,
            "--baud", "460800",
            "write-flash", "0x0", merged_bin
        ]
    else:
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
            "write-flash",
            "0x1000", bootloader,
            "0x8000", partition,
            "0x10000", receiver
        ]

    print(f"[CMD] {' '.join(cmd)}\n")
    print("⏳ جاري الحرق... (إذا علق في Connecting... اضغط باستمرار على زر BOOT في البوردة لمدة ثانيتين)\n")

    res = subprocess.run(cmd)
    if res.returncode == 0:
        print("\n✅ تم حرق فيرموير الـ Receiver بنجاح تام 100%!")
        print("   البوردة تدعم الآن الربط مع CSI_TX_NODE و CSI_NODE_EXT وتستقبل 20 فريم/ثانية.")
        return True
    else:
        print("\n❌ حدث خطأ أثناء الحرق. تأكد من إغلاق أي برنامج يستخدم المنفذ (مثل Arduino IDE أو Serial Monitor).")
        return False

def verify_ai_model():
    """Checks if trained ML model and scaler exist."""
    print("\n🔍 [فحص موديل الذكاء الاصطناعي ومعايرة المناطق]...")
    paths = [
        os.path.join(ROOT_DIR, "dashboard", "csi_model.pkl"),
        os.path.join(ROOT_DIR, "dashboard", "scaler.pkl"),
        os.path.join(ROOT_DIR, "dashboard", "feature_cols.pkl"),
    ]
    if all(os.path.exists(p) for p in paths):
        print("✅ موديل الذكاء الاصطناعي جاهز ومحمّل (RandomForest + Scaler).")
        return True
    else:
        print("⚠️ الموديل غير موجود؛ جاري تدريبه الآن تلقائياً...")
        train_script = os.path.join(ROOT_DIR, "ml", "ml_model.py")
        res = subprocess.run([sys.executable, train_script])
        return res.returncode == 0

def test_live_csi_stream(port: str):
    """Listens for CSI packets to verify hardware link."""
    print("\n" + "="*80)
    print(f"🧪 [فحص بث إشارات الـ CSI الحية من بوردة {port}]...")
    print("="*80)
    try:
        import serial
    except ImportError:
        print("❌ pyserial غير مثبت.")
        return

    try:
        ser = serial.Serial(port=port, baudrate=115200, timeout=1.5)
        print("✅ تم فتح المنفذ بنجاح. جاري الاستماع للفريمات لمدة 5 ثوانٍ...\n")
        start = time.time()
        packets = 0
        while time.time() - start < 5.0:
            if ser.in_waiting:
                line = ser.readline().decode("utf-8", errors="ignore").strip()
                if "CSI_START" in line and "CSI_END" in line:
                    packets += 1
                    parts = line.split(",")
                    rssi_val = parts[2] if len(parts) > 2 else "--"
                    subs_val = parts[3] if len(parts) > 3 else "64"
                    print(f"   [FRAME #{packets}] RSSI: {rssi_val} dBm | Subcarriers: {subs_val} | Samples: {parts[4:10]}...")
            time.sleep(0.01)
        ser.close()
        if packets > 0:
            print(f"\n🎉 رائع! تم استقبال {packets} فريم CSI بنجاح تام! الهاردوير شغال 100%.")
        else:
            print("\n⚠️ لم يتم التقاط فريمات CSI. تأكد من تشغيل بوردة الـ Transmitter (ESP32 #1).")
    except Exception as e:
        print(f"❌ تعذر فتح المنفذ: {e}")

def run_dashboard(port: str = ""):
    """Launches the Streamlit Live Dashboard."""
    print("\n" + "="*80)
    print("🚀 [إطلاق الداش بورد الحية - 3-Node Thermal Radar Dashboard]")
    print("="*80)
    app_path = os.path.join(ROOT_DIR, "dashboard", "app.py")
    
    env = os.environ.copy()
    if port:
        env["CSI_DEFAULT_PORT"] = port
        
    # Clean up any orphaned streamlit processes to avoid COM port locks
    try:
        import psutil
        curr_pid = os.getpid()
        for p in psutil.process_iter(['pid', 'name', 'cmdline']):
            if p.info['pid'] != curr_pid:
                cmd_line = ' '.join(p.info.get('cmdline') or []).lower()
                if 'streamlit' in cmd_line and 'app.py' in cmd_line:
                    try:
                        p.kill()
                    except Exception:
                        pass
        time.sleep(0.4)
    except Exception:
        pass

    print("[*] جاري تشغيل خادم Streamlit وفتح المتصفح...")
    print("[*] Dashboard URL: http://localhost:8501")
    print("--------------------------------------------------------------------------------")
    print("💡 النظام يعمل تلقائياً الآن:")
    print("   1) التتبع الحي يعمل فوراً وبدون الحاجة للضغط على أي أزرار.")
    print("   2) تحرك بين زوايا الغرفة وراقب الهالة الحرارية التجسيمية للشخص وهي تتبعك بدقة!")
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
    if not verify_ai_model():
        print("\n❌ لا يمكن تشغيل الداش بورد قبل إصلاح تدريب الموديل.")
        return
        
    # 3. Fast-Launch or Menu
    print("\n" + "="*80)
    print("🚀 [جاهز للتشغيل الفوري - 1-Click Launch]")
    print(f"   المنفذ المحدد: {selected_port}")
    print("   سيتم تشغيل الداش بورد والتتبع الحراري الثلاثي فوراً تلقائياً.")
    print("="*80)
    
    print("\nاضغط Enter للتشغيل فوراً، أو اكتب 'M' لفتح القائمة المتقدمة (حرق/فحص):")
    try:
        # Prompt with short timeout or immediate enter
        choice = input("--> ").strip().upper()
    except Exception:
        choice = ""
        
    if choice == "M":
        print("\n📋 [القائمة المتقدمة]:")
        print("   [1] تشغيل الداش بورد الحية فوراً")
        print("   [2] حرق الفيرموير على بوردة الاستقبال (ESP32 #2)")
        print("   [3] فحص واختبار قراءة فريمات الـ CSI الحية")
        print("   [4] إعادة تدريب الموديل")
        sub_choice = input("اختر رقم الخيار [1-4]: ").strip()
        if sub_choice == "2":
            flash_receiver(selected_port)
            time.sleep(2.0)
            run_dashboard(selected_port)
        elif sub_choice == "3":
            test_live_csi_stream(selected_port)
            input("\nاضغط Enter لتشغيل الداش بورد...")
            run_dashboard(selected_port)
        elif sub_choice == "4":
            subprocess.run([sys.executable, os.path.join(ROOT_DIR, "ml", "ml_model.py")])
            input("\nاضغط Enter لتشغيل الداش بورد...")
            run_dashboard(selected_port)
        else:
            run_dashboard(selected_port)
    else:
        run_dashboard(selected_port)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[!] تم إيقاف البرنامج بواسطة المستخدم.")
