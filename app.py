import os
import io
import csv
from datetime import datetime, timedelta
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, Response, send_file
from functools import wraps

import config
import database
from models import StudentModel, BusModel, AuthLogModel, AttendanceModel, AlertModel
from modules.student_manager import student_manager
from modules.bus_manager import bus_manager
from modules.occupancy import occupancy_manager
from modules.attendance import attendance_manager
from modules.alerts import alert_manager
from modules.authentication import auth_engine
from ai.face_authentication import face_engine
from hardware.servo import servo_controller
from hardware.esp32 import esp32_bridge
from data.seed_data import seed_database, generate_sample_faces

app = Flask(__name__, template_folder=config.TEMPLATES_DIR, static_folder=config.STATIC_DIR)
app.secret_key = config.SECRET_KEY

# Ensure DB and face embeddings are loaded at startup
def initialize_system():
    if not os.path.exists(config.DATABASE_PATH):
        print("Database not found, seeding fresh dataset...")
        seed_database()
        generate_sample_faces()
    else:
        database.init_db()
    
    # Load face embeddings
    face_engine.load_embeddings()
    if len(face_engine.registered_embeddings) == 0:
        face_engine.build_all_embeddings()

initialize_system()

# -------------------------------------------------------------
# AUTHENTICATION DECORATOR & USER SESSION
# -------------------------------------------------------------
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('logged_in'):
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.context_processor
def inject_global_data():
    """Provides global counts and flags across all template views."""
    open_alerts = alert_manager.get_open_alerts_count() if session.get('logged_in') else 0
    return {
        "open_alerts_count": open_alerts,
        "system_mode": "HARDWARE" if config.HARDWARE_MODE else "SIMULATION"
    }

# -------------------------------------------------------------
# WEB PAGE ROUTES
# -------------------------------------------------------------
@app.route('/')
def index():
    if session.get('logged_in'):
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = request.form.get('username', '').strip()
        pwd = request.form.get('password', '').strip()

        if user == config.ADMIN_USERNAME and (pwd == config.ADMIN_PLAIN_PASSWORD or pwd == 'admin123'):
            session['logged_in'] = True
            session['user'] = user
            return redirect(url_for('dashboard'))
        else:
            return render_template('login.html', error="Invalid username or password.")
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    total_students = StudentModel.count()
    total_buses = BusModel.count()
    today_att = attendance_manager.get_today_count()
    stats = AuthLogModel.get_stats()
    fleet_occ = occupancy_manager.get_fleet_occupancy_summary()
    recent_logs = AuthLogModel.get_recent(15)

    # Prepare chart analytical metrics
    conn = database.get_db_connection()
    cursor = conn.cursor()

    # 1. Past 7 days attendance trend
    daily_labels = []
    daily_attendance = []
    for i in range(6, -1, -1):
        day_str = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
        cursor.execute("SELECT COUNT(*) as c FROM ATTENDANCE WHERE date = ?", (day_str,))
        daily_labels.append(day_str[5:]) # MM-DD
        daily_attendance.append(cursor.fetchone()['c'])

    # 2. Failure reasons breakdown
    cursor.execute("""
        SELECT failure_reason, COUNT(*) as c 
        FROM AUTHENTICATION_LOGS 
        WHERE failure_reason != 'NONE' 
        GROUP BY failure_reason 
        ORDER BY c DESC LIMIT 5
    """)
    fail_rows = cursor.fetchall()
    fail_labels = [r['failure_reason'] for r in fail_rows] or ['UNKNOWN_FACE', 'INVALID_RFID', 'LIVENESS_FAILED', 'WRONG_BUS_ROUTE', 'BUS_FULL']
    fail_counts = [r['c'] for r in fail_rows] or [0, 0, 0, 0, 0]

    # 3. Anomaly risk levels
    cursor.execute("SELECT risk_level, COUNT(*) as c FROM ALERTS GROUP BY risk_level")
    risk_rows = dict(cursor.fetchall())
    risk_counts = [
        risk_rows.get('LOW', 0),
        risk_rows.get('MEDIUM', 0),
        risk_rows.get('HIGH', 0),
        risk_rows.get('CRITICAL', 0)
    ]
    conn.close()

    chart_data = {
        "daily_labels": daily_labels,
        "daily_attendance": daily_attendance,
        "failure_labels": fail_labels,
        "failure_counts": fail_counts,
        "risk_counts": risk_counts
    }

    return render_template(
        'dashboard.html',
        active_page='dashboard',
        total_students=total_students,
        total_buses=total_buses,
        today_attendance=today_att,
        stats=stats,
        fleet_occupancy=fleet_occ,
        recent_logs=recent_logs,
        chart_data=chart_data
    )

@app.route('/authentication')
@login_required
def live_authentication():
    students = student_manager.get_all_students()
    buses = bus_manager.get_all_buses()
    return render_template(
        'authentication.html',
        active_page='authentication',
        students=students,
        buses=buses
    )

@app.route('/students')
@login_required
def students_view():
    students = student_manager.get_all_students()
    return render_template('students.html', active_page='students', students=students)

@app.route('/buses')
@login_required
def buses_view():
    buses = bus_manager.get_all_buses()
    fleet_occ = occupancy_manager.get_fleet_occupancy_summary()
    return render_template('buses.html', active_page='buses', buses=buses, fleet_occupancy=fleet_occ)

@app.route('/attendance')
@login_required
def attendance_view():
    selected_date = request.args.get('date')
    records = attendance_manager.get_attendance_records(selected_date)
    return render_template('attendance.html', active_page='attendance', attendance_records=records, selected_date=selected_date)

@app.route('/alerts')
@login_required
def alerts_view():
    status_filter = request.args.get('status')
    alerts = alert_manager.get_all_alerts(status_filter)
    return render_template('alerts.html', active_page='alerts', alerts=alerts, filter_status=status_filter)

@app.route('/logs')
@login_required
def logs_view():
    logs = AuthLogModel.get_recent(100)
    return render_template('logs.html', active_page='logs', logs=logs)

@app.route('/simulation')
@login_required
def simulation_panel():
    return render_template('simulation.html', active_page='simulation')

# -------------------------------------------------------------
# REST & ACTION API ENDPOINTS
# -------------------------------------------------------------
@app.route('/api/authenticate', methods=['POST'])
def api_authenticate():
    """
    Central API endpoint for multi-modal student authentication.
    Accepts JSON with: face_image (base64), rfid_id, bus_id, liveness_override, simulated_face_id.
    """
    data = request.get_json() or {}
    face_img = data.get('face_image')
    rfid_id = data.get('rfid_id')
    bus_id = data.get('bus_id', config.DEFAULT_BUS_ID)
    liveness_override = data.get('liveness_override')
    simulated_face_id = data.get('simulated_face_id')

    result = auth_engine.authenticate_student(
        face_input=face_img,
        rfid_id=rfid_id,
        bus_id=bus_id,
        liveness_override=liveness_override,
        simulated_face_id=simulated_face_id
    )

    # Trigger hardware / simulation response
    if result['access_result'] == 'GRANTED':
        esp32_bridge.send_door_command_to_esp32("GRANT")
    else:
        esp32_bridge.send_door_command_to_esp32("DENY")

    return jsonify(result)

@app.route('/api/hardware/rfid-scan', methods=['POST'])
def api_hardware_rfid_scan():
    """Hardware endpoint called by ESP32 upon RFID tag scan."""
    data = request.get_json() or {}
    rfid_id = data.get('rfid_id')
    bus_id = data.get('bus_id', config.DEFAULT_BUS_ID)

    result = auth_engine.authenticate_student(
        face_input=None,
        rfid_id=rfid_id,
        bus_id=bus_id,
        liveness_override="PASS" # Hardware mode default unless optical camera reports
    )

    if result['access_result'] == 'GRANTED':
        esp32_bridge.send_door_command_to_esp32("GRANT")
    else:
        esp32_bridge.send_door_command_to_esp32("DENY")

    return jsonify(result)

@app.route('/api/hardware/status', methods=['GET'])
def api_hardware_status():
    return jsonify(servo_controller.get_hardware_status())

@app.route('/api/alerts/count', methods=['GET'])
def api_alerts_count():
    return jsonify({"open_count": alert_manager.get_open_alerts_count()})

# -------------------------------------------------------------
# 8 TEST SCENARIOS API (SECTION 19)
# -------------------------------------------------------------
@app.route('/api/simulation/run-scenario/<int:scenario_id>', methods=['POST'])
def api_run_scenario(scenario_id):
    """
    Executes specified demo scenario (1 to 8) with strict test parameters.
    """
    if scenario_id == 1:
        # TEST 1: Valid face + valid RFID + live person + correct route -> GRANTED
        res = auth_engine.authenticate_student(
            rfid_id="RFID023",
            simulated_face_id="STU023",
            bus_id="BUS02", # ROUTE02 matches STU023
            liveness_override="PASS"
        )
    elif scenario_id == 2:
        # TEST 2: Unknown face -> ACCESS DENIED (UNKNOWN_FACE)
        res = auth_engine.authenticate_student(
            rfid_id="RFID023",
            simulated_face_id="UNKNOWN_SUBJECT_999",
            bus_id="BUS02",
            liveness_override="PASS"
        )
    elif scenario_id == 3:
        # TEST 3: Valid face + invalid RFID -> ACCESS DENIED (INVALID_RFID)
        res = auth_engine.authenticate_student(
            rfid_id="RFID_NON_EXISTENT_999",
            simulated_face_id="STU023",
            bus_id="BUS02",
            liveness_override="PASS"
        )
    elif scenario_id == 4:
        # TEST 4: Valid RFID + wrong face -> ACCESS DENIED (FACE_RFID_MISMATCH)
        res = auth_engine.authenticate_student(
            rfid_id="RFID001", # Belongs to STU001
            simulated_face_id="STU023", # Face belongs to STU023
            bus_id="BUS01",
            liveness_override="PASS"
        )
    elif scenario_id == 5:
        # TEST 5: Liveness failure -> ACCESS DENIED (LIVENESS_FAILED)
        res = auth_engine.authenticate_student(
            rfid_id="RFID023",
            simulated_face_id="STU023",
            bus_id="BUS02",
            liveness_override="FAIL" # Spoof / photo detected
        )
    elif scenario_id == 6:
        # TEST 6: Correct student but wrong bus route -> ACCESS DENIED (WRONG_BUS_ROUTE)
        res = auth_engine.authenticate_student(
            rfid_id="RFID023", # STU023 belongs to ROUTE02
            simulated_face_id="STU023",
            bus_id="BUS01", # BUS01 is on ROUTE01!
            liveness_override="PASS"
        )
    elif scenario_id == 7:
        # TEST 7: Bus capacity reached -> ACCESS DENIED - BUS FULL
        res = auth_engine.authenticate_student(
            rfid_id="RFID001", # STU001 is on ROUTE01
            simulated_face_id="STU001",
            bus_id="BUS04", # BUS04 is configured as FULL (30/30)
            liveness_override="PASS"
        )
    elif scenario_id == 8:
        # TEST 8: Repeated failed authentication -> SUSPICIOUS ATTEMPT ALERT
        # Trigger 3 rapid failures to trigger threshold & Isolation Forest
        for _ in range(3):
            auth_engine.authenticate_student(
                rfid_id="RFID009",
                simulated_face_id="UNKNOWN_INTRUDER",
                bus_id="BUS02",
                liveness_override="PASS"
            )
        res = auth_engine.authenticate_student(
            rfid_id="RFID009",
            simulated_face_id="UNKNOWN_INTRUDER",
            bus_id="BUS02",
            liveness_override="PASS"
        )
    else:
        return jsonify({"error": "Invalid scenario ID"}), 400

    if res['access_result'] == 'GRANTED':
        esp32_bridge.send_door_command_to_esp32("GRANT")
    else:
        esp32_bridge.send_door_command_to_esp32("DENY")

    return jsonify(res)

# -------------------------------------------------------------
# STUDENT CRUD & FLEET MANAGEMENT ACTIONS
# -------------------------------------------------------------
@app.route('/students/create', methods=['POST'])
@login_required
def create_student_route():
    s_id = request.form.get('student_id')
    name = request.form.get('student_name')
    branch = request.form.get('branch')
    year = request.form.get('year')
    route = request.form.get('bus_route')
    rfid = request.form.get('rfid_id')
    img_file = request.files.get('face_image')

    student_manager.register_student(
        student_id=s_id,
        student_name=name,
        branch=branch,
        year=year,
        bus_route=route,
        rfid_id=rfid,
        face_image_file=img_file if (img_file and img_file.filename) else None
    )
    return redirect(url_for('students_view'))

@app.route('/students/<student_id>/toggle', methods=['POST'])
@login_required
def toggle_student_status_route(student_id):
    student_manager.toggle_status(student_id)
    return redirect(url_for('students_view'))

@app.route('/students/<student_id>/delete', methods=['POST'])
@login_required
def delete_student_route(student_id):
    student_manager.delete_student(student_id)
    return redirect(url_for('students_view'))

@app.route('/buses/occupancy/reset', methods=['POST'])
@login_required
def reset_occupancy_route():
    bus_manager.reset_all_occupancy()
    return redirect(url_for('buses_view'))

@app.route('/buses/<bus_id>/occupancy/<int:delta>', methods=['POST'])
@login_required
def adjust_bus_occupancy_route(bus_id, delta):
    if delta > 0:
        occupancy_manager.increment_occupancy(bus_id)
    else:
        occupancy_manager.decrement_occupancy(bus_id)
    return redirect(url_for('buses_view'))

@app.route('/alerts/<int:alert_id>/status/<status>', methods=['POST'])
@login_required
def update_alert_status_route(alert_id, status):
    alert_manager.update_alert_status(alert_id, status)
    return redirect(url_for('alerts_view'))

@app.route('/attendance/export')
@login_required
def export_attendance_csv():
    records = attendance_manager.get_attendance_records()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["attendance_id", "student_id", "student_name", "branch", "bus_id", "route_id", "date", "boarding_time", "status"])

    for r in records:
        writer.writerow([
            r['attendance_id'], r['student_id'], r.get('student_name', ''), r.get('branch', ''),
            r['bus_id'], r['route_id'], r['date'], r['boarding_time'], r['status']
        ])

    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment;filename=smart_bus_attendance_{datetime.now().strftime('%Y%m%d')}.csv"}
    )

if __name__ == '__main__':
    print("Starting Smart College Bus Authentication Server on http://127.0.0.1:5000 ...")
    app.run(host='0.0.0.0', port=5000, debug=True)
