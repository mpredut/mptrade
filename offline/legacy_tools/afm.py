"""A historical manual utility; it is not part of the runtime."""

from plyer import notification
import requests
import hashlib
import time
import utils  # Presupunem ca utils.py contine functia beep

# A function that generates the page's hash.
def get_page_hash(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            page_content = response.content
            return hashlib.md5(page_content).hexdigest()
        else:
            print(f"Could not reach the page. Status code: {response.status_code}")
            return None
    except Exception as e:
        print(f"Eroare la accesarea paginii: {e}")
        return None

# A function that shows a cross-platform notification.
def show_notification(title, text):
    notification.notify(
        title=title,
        message=text,
        timeout=10  # Durata notificarii in secunde
    )

# A function that plays the sound through utils.beep.
def play_sound():
    try:
        utils.beep(3)  # We call the beep function from utils.py.
    except Exception as e:
        print(f"Eroare la redarea sunetului: {e}")

if __name__ == "__main__":
    # URL-ul paginii
    url = 'https://depunerefotovoltaice.afm.ro/'

    # Hash-ul initial al paginii
    last_hash = get_page_hash(url)

    if last_hash:
        print("Monitorizare pornita...")

        while True:
            time.sleep(1)  # Check every second.
            current_hash = get_page_hash(url)
            print("Nothing")
            if current_hash and current_hash != last_hash:
                print("Pagina s-a schimbat!............................................HAHA!")
                play_sound()
                last_hash = current_hash  # Actualizeaza hash-ul paginii
                break
            elif not current_hash:
                print("Eroare la preluarea paginii. Reincercam...")
