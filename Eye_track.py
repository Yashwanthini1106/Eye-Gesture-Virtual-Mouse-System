import cv2
import mediapipe as mp
import pyautogui
import time

pyautogui.FAILSAFE = False


# =========================================================
# CAMERA
# =========================================================

cam = cv2.VideoCapture(0)

face_mesh = mp.solutions.face_mesh.FaceMesh(
    refine_landmarks=True,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

screen_w, screen_h = pyautogui.size()


# =========================================================
# BLINK SETTINGS
# =========================================================

BLINK_THRESHOLD = 0.012

MIN_BLINK = 0.05
MAX_BLINK = 0.40

# Long LEFT eye close = Text Selection
# Long RIGHT eye close = Drag & Drop
LONG_BLINK = 0.65

CLICK_COOLDOWN = 0.60

# Double both-eye blink
DOUBLE_BLINK_TIME = 1.2


# =========================================================
# SCROLL SETTINGS
# =========================================================

HEAD_MOVE_THRESHOLD = 0.012

SCROLL_AMOUNT = 5

SCROLL_TIMEOUT = 5.0

HEAD_SMOOTHING = 0.25


# =========================================================
# CURSOR SETTINGS
# =========================================================

SMOOTHING = 0.25


# =========================================================
# LEFT EYE
# =========================================================

left_eye_closed = False
left_blink_start = 0

last_left_click = 0


# =========================================================
# RIGHT EYE
# =========================================================

right_eye_closed = False
right_blink_start = 0

last_right_click = 0


# =========================================================
# TEXT SELECTION
# =========================================================

text_selecting = False

selection_just_finished = False
selection_finished_time = 0


# =========================================================
# DRAG & DROP
# =========================================================

dragging = False


# =========================================================
# DOUBLE BLINK
# =========================================================

both_eyes_closed = False
both_blink_start = 0

blink_count = 0
first_blink_time = 0


# =========================================================
# SCROLL MODE
# =========================================================

scroll_mode = False

scroll_reference_y = None
scroll_start_time = 0

scroll_smoothed_y = None


# =========================================================
# CURSOR
# =========================================================

smooth_x = screen_w / 2
smooth_y = screen_h / 2


# =========================================================
# HELPER
# =========================================================

def eye_distance(landmarks, top, bottom):

    return abs(
        landmarks[top].y -
        landmarks[bottom].y
    )


# =========================================================
# STOP TEXT SELECTION
# =========================================================

def stop_text_selection():

    global text_selecting
    global selection_just_finished
    global selection_finished_time

    if text_selecting:

        pyautogui.mouseUp(
            button="left"
        )

        text_selecting = False

        selection_just_finished = True

        selection_finished_time = time.time()

        print("==============================")
        print("TEXT SELECTION FINISHED")
        print("==============================")


# =========================================================
# STOP DRAGGING / DROP
# =========================================================

def stop_dragging():

    global dragging

    if dragging:

        pyautogui.mouseUp(
            button="left"
        )

        dragging = False

        print("==============================")
        print("DROP COMPLETED")
        print("==============================")


# =========================================================
# START SCROLL MODE
# =========================================================

def start_scroll_mode(head_position):

    global scroll_mode
    global scroll_reference_y
    global scroll_start_time

    # Safely stop text selection
    if text_selecting:

        stop_text_selection()

    # Safely stop dragging
    if dragging:

        stop_dragging()

    scroll_mode = True

    scroll_reference_y = head_position

    scroll_start_time = time.time()

    print("==============================")
    print("DOUBLE BLINK DETECTED")
    print("SCROLL MODE ON")
    print(
        f"REFERENCE HEAD: {head_position:.4f}"
    )
    print("HEAD UP   = SCROLL UP")
    print("HEAD DOWN = SCROLL DOWN")
    print("==============================")


# =========================================================
# MAIN LOOP
# =========================================================

while True:

    success, frame = cam.read()

    if not success:
        break


    # =====================================================
    # MIRROR CAMERA
    # =====================================================

    frame = cv2.flip(
        frame,
        1
    )


    # =====================================================
    # RGB
    # =====================================================

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # =====================================================
    # MEDIAPIPE
    # =====================================================

    output = face_mesh.process(
        rgb_frame
    )

    current_time = time.time()


    # =====================================================
    # FACE FOUND
    # =====================================================

    if output.multi_face_landmarks:

        landmarks = (
            output.multi_face_landmarks[0].landmark
        )


        # =================================================
        # CURSOR
        # =================================================

        iris = landmarks[475]

        screen_x = (
            screen_w *
            iris.x
        )

        screen_y = (
            screen_h *
            iris.y
        )


        # Cursor range expansion

        screen_x = (
            (screen_x - screen_w / 2)
            * 1.3
            + screen_w / 2
        )

        screen_y = (
            (screen_y - screen_h / 2)
            * 1.3
            + screen_h / 2
        )


        # Keep inside screen

        screen_x = max(
            0,
            min(
                screen_w - 1,
                screen_x
            )
        )

        screen_y = max(
            0,
            min(
                screen_h - 1,
                screen_y
            )
        )


        # =================================================
        # CURSOR SMOOTHING
        # =================================================

        smooth_x = (
            smooth_x *
            (1 - SMOOTHING)
            +
            screen_x *
            SMOOTHING
        )

        smooth_y = (
            smooth_y *
            (1 - SMOOTHING)
            +
            screen_y *
            SMOOTHING
        )


        # =================================================
        # MOVE CURSOR
        # =================================================

        pyautogui.moveTo(
            int(smooth_x),
            int(smooth_y),
            duration=0.01
        )


        # =================================================
        # EYE DISTANCE
        # =================================================

        left_eye_distance = eye_distance(
            landmarks,
            159,
            145
        )

        right_eye_distance = eye_distance(
            landmarks,
            386,
            374
        )


        # =================================================
        # EYE STATUS
        # =================================================

        left_closed = (
            left_eye_distance <
            BLINK_THRESHOLD
        )

        right_closed = (
            right_eye_distance <
            BLINK_THRESHOLD
        )


        # =================================================
        # HEAD POSITION
        # =================================================

        nose_y = landmarks[1].y

        left_eye_center = (
            landmarks[159].y +
            landmarks[145].y
        ) / 2

        right_eye_center = (
            landmarks[386].y +
            landmarks[374].y
        ) / 2

        eye_center_y = (
            left_eye_center +
            right_eye_center
        ) / 2

        raw_head_y = (
            nose_y -
            eye_center_y
        )


        # =================================================
        # HEAD SMOOTHING
        # =================================================

        if scroll_smoothed_y is None:

            scroll_smoothed_y = raw_head_y

        else:

            scroll_smoothed_y = (
                scroll_smoothed_y *
                (1 - HEAD_SMOOTHING)
                +
                raw_head_y *
                HEAD_SMOOTHING
            )

        head_position_y = scroll_smoothed_y


        # =================================================
        # BOTH EYES CLOSED
        # =================================================

        if left_closed and right_closed:

            if not both_eyes_closed:

                both_eyes_closed = True

                both_blink_start = current_time


        # =================================================
        # BOTH EYES OPENED
        # =================================================

        elif both_eyes_closed:

            both_eyes_closed = False

            duration = (
                current_time -
                both_blink_start
            )


            if (
                MIN_BLINK
                <= duration
                <= MAX_BLINK
            ):

                # FIRST BLINK

                if blink_count == 0:

                    blink_count = 1

                    first_blink_time = current_time

                    print(
                        "FIRST BOTH-EYE BLINK"
                    )


                # SECOND BLINK

                elif (
                    current_time -
                    first_blink_time
                    <= DOUBLE_BLINK_TIME
                ):

                    blink_count = 0

                    start_scroll_mode(
                        head_position_y
                    )


        # =================================================
        # DOUBLE BLINK TIMEOUT
        # =================================================

        if blink_count == 1:

            if (
                current_time -
                first_blink_time
                > DOUBLE_BLINK_TIME
            ):

                blink_count = 0


        # =================================================
        # SCROLL MODE
        # =================================================

        if scroll_mode:

            current_head_y = head_position_y

            movement = (
                current_head_y -
                scroll_reference_y
            )


            cv2.putText(
                frame,
                f"HEAD MOVE: {movement:.4f}",
                (20, 155),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 0),
                2
            )


            # HEAD DOWN

            if (
                movement >
                HEAD_MOVE_THRESHOLD
            ):

                pyautogui.scroll(
                    -SCROLL_AMOUNT
                )

                print(
                    f"SCROLL DOWN | "
                    f"movement={movement:.4f}"
                )

                scroll_mode = False

                scroll_reference_y = None


            # HEAD UP

            elif (
                movement <
                -HEAD_MOVE_THRESHOLD
            ):

                pyautogui.scroll(
                    SCROLL_AMOUNT
                )

                print(
                    f"SCROLL UP | "
                    f"movement={movement:.4f}"
                )

                scroll_mode = False

                scroll_reference_y = None


            # TIMEOUT

            elif (
                current_time -
                scroll_start_time
                > SCROLL_TIMEOUT
            ):

                scroll_mode = False

                scroll_reference_y = None

                print(
                    "SCROLL MODE TIMEOUT"
                )


            cv2.putText(
                frame,
                "SCROLL MODE",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2
            )


        # =================================================
        # LEFT EYE ONLY
        # =================================================

        if (
            left_closed
            and
            not right_closed
        ):

            if not left_eye_closed:

                left_eye_closed = True

                left_blink_start = current_time


        # =================================================
        # LEFT EYE OPENED
        # =================================================

        elif (
            not left_closed
            and
            left_eye_closed
        ):

            left_eye_closed = False

            duration = (
                current_time -
                left_blink_start
            )


            # =================================================
            # LONG LEFT EYE CLOSE
            # TEXT SELECTION
            # =================================================

            if duration >= LONG_BLINK:

                if not text_selecting:

                    if (
                        not scroll_mode
                        and
                        not dragging
                    ):

                        pyautogui.mouseDown(
                            button="left"
                        )

                        text_selecting = True

                        selection_just_finished = False

                        print("==============================")
                        print("TEXT SELECTION START")
                        print("MOVE EYES TO SELECT TEXT")
                        print("LONG LEFT EYE CLOSE AGAIN")
                        print("TO FINISH")
                        print("==============================")


                else:

                    stop_text_selection()


            # =================================================
            # NORMAL LEFT EYE CLOSE
            # LEFT CLICK
            # =================================================

            elif (
                MIN_BLINK
                <= duration
                <= MAX_BLINK
            ):

                if (
                    not text_selecting
                    and
                    not dragging
                    and
                    not scroll_mode
                ):

                    if (
                        current_time -
                        selection_finished_time
                        > 0.30
                    ):

                        if (
                            current_time -
                            last_left_click
                            > CLICK_COOLDOWN
                        ):

                            pyautogui.click(
                                button="left"
                            )

                            print(
                                "LEFT CLICK"
                            )

                            last_left_click = current_time


        # =================================================
        # RIGHT EYE ONLY
        # =================================================

        if (
            right_closed
            and
            not left_closed
        ):

            if not right_eye_closed:

                right_eye_closed = True

                right_blink_start = current_time


        # =================================================
        # RIGHT EYE OPENED
        # =================================================

        elif (
            not right_closed
            and
            right_eye_closed
        ):

            right_eye_closed = False

            duration = (
                current_time -
                right_blink_start
            )


            # =================================================
            # LONG RIGHT EYE CLOSE
            # DRAG & DROP
            # =================================================

            if duration >= LONG_BLINK:

                # ---------------------------------------------
                # START DRAG
                # ---------------------------------------------

                if not dragging:

                    if (
                        not scroll_mode
                        and
                        not text_selecting
                    ):

                        pyautogui.mouseDown(
                            button="left"
                        )

                        dragging = True

                        print("==============================")
                        print("RIGHT EYE LONG CLOSE")
                        print("DRAG STARTED")
                        print("MOVE EYES TO DESTINATION")
                        print("RIGHT EYE LONG CLOSE AGAIN")
                        print("DROP")
                        print("==============================")


                # ---------------------------------------------
                # DROP
                # ---------------------------------------------

                else:

                    stop_dragging()


            # =================================================
            # NORMAL RIGHT EYE CLOSE
            # RIGHT CLICK
            # =================================================

            elif (
                MIN_BLINK
                <= duration
                <= MAX_BLINK
            ):

                if (
                    not dragging
                    and
                    not text_selecting
                    and
                    not scroll_mode
                ):

                    if (
                        current_time -
                        last_right_click
                        > CLICK_COOLDOWN
                    ):

                        pyautogui.click(
                            button="right"
                        )

                        print(
                            "RIGHT CLICK"
                        )

                        last_right_click = current_time


        # =================================================
        # DEBUG
        # =================================================

        cv2.putText(
            frame,
            f"L: {left_eye_distance:.3f}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"R: {right_eye_distance:.3f}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"DOUBLE: {blink_count}",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2
        )


        # =================================================
        # STATUS DISPLAY
        # =================================================

        if dragging:

            cv2.putText(
                frame,
                "DRAGGING...",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        elif text_selecting:

            cv2.putText(
                frame,
                "TEXT SELECTING...",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        elif selection_just_finished:

            cv2.putText(
                frame,
                "ACTION FINISHED",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )


    # =====================================================
    # DISPLAY
    # =====================================================

    cv2.imshow(
        "Eye Control",
        frame
    )


    # =====================================================
    # ESC
    # =====================================================

    if cv2.waitKey(1) & 0xFF == 27:
        break


# =========================================================
# CLEANUP
# =========================================================

if text_selecting:

    pyautogui.mouseUp(
        button="left"
    )

    text_selecting = False


if dragging:

    pyautogui.mouseUp(
        button="left"
    )

    dragging = False


cam.release()

face_mesh.close()

cv2.destroyAllWindows()
