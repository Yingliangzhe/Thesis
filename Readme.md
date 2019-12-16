# Verwendung von Programm

### FOUP_DETECTION.py

Der Skript dient dazu, dass der FOUP-Flansch lokalisiert werden kann. 



### PIN_DETECTION.py

Für Pindetektion. 



### createChessboardPoint.py

Nur für die Erzeugung von 3D Objektpunkten



### calibrate.py

Kamerakalibrierung. 

Der Prozess wird zwei mal durchgeführt. 

1. Mal wurde die intrinsischen Parameter berechnet 
2. Mal nutzt die intrinsischen Parameter, um die extrinsischen Parameter zu bestimmen.

Jedesmal wird die Überprüfung der Kamerakalibrierung durch reprojection gemacht. 

Außerdem werden die Kamera Parameter von den zwei Mals gespeichert.



### solvepnpTest.py

Der Skript vorbereitet die Ergebnisse aus *solvPnP* und Roboterbahn. Die Daten werden in 

````
calib_pnp_rotvec.npy
calib_pnp_tvec.npy

calib_calibration_rotvec.npy
calib_calibration_tvec.npy
````

abgespeichert. 

Aus den Daten kann der Algorithmus *solvePnP* velidiert werden. 



### Calibration_PnP_Robot.py

Der Skript stellt den Vergleich von *solvePnP* und Roboterbahn dar. Die Daten sind aus (solvepnpTest.py). 



### Calibration_FOUP_Robot.py

Der Skript vergleicht das Ergebnis von FOUP Lokalisierungsalgorithmus mit Schachbrett Lokalisierungsalgorithmus sowie Referenzpunkte in Roboterbahn. 



