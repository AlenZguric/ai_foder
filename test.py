import cv2

# Učitaj sliku
image = cv2.imread("return.jpg", cv2.IMREAD_GRAYSCALE)

# Kreiraj ORB objekat
orb = cv2.ORB_create()

# Detekcija ključnih tačaka i deskriptora
keypoints, descriptors = orb.detectAndCompute(image, None)

print(f"Ključne tačke: {len(keypoints)}")
