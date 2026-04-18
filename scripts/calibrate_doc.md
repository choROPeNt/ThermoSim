# Kamerakalibrierung — `calibrate.py`

**Ziel:** Kamerakalibrierung einer IR-Thermalkamera anhand von Schachbrettmustern
aus transienten Thermogrammen.

---

## Pipeline pro Frame

- **Flat-field-Korrektur** — teilt jeden Pixel durch eine stark geglättete Version
  des Bildes (Hintergrundschätzung via Gauss-Blur). Entfernt langsame
  Temperaturgradienten im Hintergrund und erhöht den lokalen Kontrast,
  damit das Schachbrett hervorsticht.

- **Median-Filter** — entfernt isolierte Hotspots/Kaltspots (Impulsrauschen)
  vor der Kantendetektion, ohne die Schachbrettkanten zu verschmieren.

- **Canny-Kantendetektion** — findet scharfe Intensitätsgradienten im
  vorverarbeiteten Bild; liefert eine binäre Kantenkarte.

- **Zuschneiden (Crop)** — berechnet die minimale Bounding Box um alle
  gefundenen Kanten mit optionalem Padding, um den Bildbereich auf das
  relevante Muster zu reduzieren.

- **Otsu-Schwellwert** — automatische Binarisierung des zugeschnittenen
  flat-field-Bildes; teilt das Bild in zwei Klassen (hell/dunkel) durch
  Maximierung der Varianz zwischen den Klassen — ideal für Schachbrettmuster.

- **Schachbrett-Eckenerkennung** (`findChessboardCornersSB`) — OpenCVs
  Sub-Pixel-genauer Detektor für Schachbrettecken; wird bei Misserfolg mit
  invertiertem Bild nochmals versucht.

---

## Kalibrierung

- Aus allen Frames mit gefundenen Ecken werden 2D-Bildpunkte und
  3D-Objektpunkte gesammelt.

- `cv2.calibrateCamera` schätzt die intrinsische **Kameramatrix K**
  (Brennweite, Hauptpunkt) und **Verzeichnungskoeffizienten**
  (radiale + tangentiale Linsenverzerrung).

- Der **Reprojektionsfehler** (in Pixel) gibt an, wie gut die geschätzte
  Kamera die bekannten 3D-Punkte auf die gemessenen 2D-Punkte abbildet —
  typisch < 1 px gilt als gut.
