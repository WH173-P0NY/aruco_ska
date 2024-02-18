import cv2
import numpy as np

# Funkcja do dzielenia obrazu na dwie równe części
def podziel_obraz(obraz):
    srodek = obraz.shape[1] // 2
    lewa_czesc = obraz[:, :srodek]
    prawa_czesc = obraz[:, srodek:]
    return lewa_czesc, prawa_czesc

# Funkcja do wykrywania znaczników ArUco
def wykryj_znaczniki(obraz):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    parametry = cv2.aruco.DetectorParameters_create()
    rogowe_punkty, id_znacznikow, _ = cv2.aruco.detectMarkers(obraz, aruco_dict, parameters=parametry)
    return rogowe_punkty, id_znacznikow

# Funkcja do obliczania odległości znaczników od kamery
def oblicz_odleglosc(f, d, delta_x):
    if delta_x == 0:
        return float('inf')  # Aby uniknąć dzielenia przez zero
    return f * d / delta_x

# Inicjalizacja kamery
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Nie można otworzyć kamery.")
    exit()

# Stałe
rozstaw_kamer = 750.0  # mm
ogniskowa_mm = 1.69  # mm, przykładowa ogniskowa w mm
szerokosc_sensora_mm = 36  # mm, szerokość matrycy
ogniskowa_px = ogniskowa_mm * (1280 / szerokosc_sensora_mm)  # przeliczenie ogniskowej na piksele

while True:
    # Czytanie pojedynczej klatki z kamery
    ret, frame = cap.read()
    if not ret:
        print("Error 2137")
        break

    # Dzielenie obrazu na dwie części
    lewa_czesc, prawa_czesc = podziel_obraz(frame)

    # Wykrywanie znaczników ArUco w obu częściach
    punkty_lewa, id_znacznikow_lewa = wykryj_znaczniki(lewa_czesc)
    punkty_prawa, id_znacznikow_prawa = wykryj_znaczniki(prawa_czesc)

    # Szukanie pasujących ID znaczników
    if id_znacznikow_lewa is not None and id_znacznikow_prawa is not None:
        wspolne_id = np.intersect1d(id_znacznikow_lewa, id_znacznikow_prawa)
        for id_znacznika in wspolne_id:
            index_lewy = np.where(id_znacznikow_lewa == id_znacznika)[0][0]
            index_prawy = np.where(id_znacznikow_prawa == id_znacznika)[0][0]
            srodek_lewy = np.mean(punkty_lewa[index_lewy][0], axis=0)
            srodek_prawy = np.mean(punkty_prawa[index_prawy][0], axis=0)
            delta_x = np.abs(srodek_lewy[0] - srodek_prawy[0])
            odleglosc = oblicz_odleglosc(ogniskowa_px, rozstaw_kamer, delta_x)
            print(f"ID znacznika: {id_znacznika}, Odległość: {odleglosc:.2f} mm")

    cv2.imshow('Obraz z kamery', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
