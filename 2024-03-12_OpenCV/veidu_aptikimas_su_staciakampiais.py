import cv2
import pandas as pd
import os

from matplotlib import pyplot as plt


def find_faces(img_path):
    # Įkeliame vaizdų atpažinimo modelį HAAR:
    # HAAR modelis ~2005 m. pristatytas; jis yra paprastas, greitas ir efektyvus naudojimui realiu laiku, vis dar populiarus šiandien,
    # ieško retimų paveikslo vietų intensyvumo skirtumų
    # trūkumai: smarkiai priklauso nuo apšvietimo sąlygų, veido pasisukimo
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # Įkeliame vaizdą ir pakeičiame į pilkus atspalvius
    img = cv2.imread(img_path)
    # Jei paveikslas ir taip nespalvotas, tai šita komanda gali paveikslą dar patamsinti
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Ieškome veidų paveiksliuke
    faces = face_cascade.detectMultiScale(gray,
                                          scaleFactor=1.1, # mažinimo dydis kiekviename mastelio lygyje
                                          minNeighbors=7, # kiek etapų kiekvienas kanditatas turi praeiti
                                          minSize=(20, 20)) # minimalus paveikslo dydis pikseliais (pvz., jei minia iš viršaus fotografuojama)

    for (x, y, w, h) in faces:
        cv2.rectangle(
            img,
            (x,y), (x+w, y+h),
            (0,0,255), # spalvos (mėlyna, žalia, raudona)
            2) # linijos storis 2
    cv2.imshow(img_path, img)
    cv2.waitKey(10000)
    cv2.destroyAllWindows()
    return faces



csv_rinkmena = 'veidai.csv'
images_folder = 'veidukai'
data = []

# files = [ f for f in os.listdir(images_folder) if f.endswith('jpg') or f.endswith('png')]
# print(files)
# for f in files:
#     veidai = find_faces(images_folder + '/' + f)
#     print(veidai)

if 0: #os.path.exists(csv_rinkmena):
    df = pd.read_csv(csv_rinkmena)
else:
    for filename in os.listdir(images_folder):
        if filename.endswith('jpg') or filename.endswith('png'):
            img_path = os.path.join(images_folder, filename)
            faces = find_faces(img_path)
            data.append({'filename': filename, 'faces_count': len(faces)})
    df = pd.DataFrame(data)
    df.to_csv(csv_rinkmena)

print(df)
print()

average_faces = df['faces_count'].mean()
print(f'Vidutinis veidų skaičius:', average_faces)

max_faces = df.loc[df['faces_count'].idxmax()]
print(f"Daugiausia veidų yra paveiksle: {max_faces['filename']}, veidų: {max_faces['faces_count']}")

df['faces_count'].plot(kind='hist', title='Veidų skaičiaus pasiskirstymas', bins=10)
plt.xlabel('Veidų skaičius')
plt.ylabel('Paveikslų skaičius')
plt.show()