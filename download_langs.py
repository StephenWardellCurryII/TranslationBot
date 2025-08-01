# download_langs.py
from argostranslate import package, translate

# Step 1: Update the package index
package.update_package_index()

# Step 2: Get available packages
available_packages = package.get_available_packages()

# Step 3: Filter and download en→hi and en→bn
for pkg in available_packages:
    if (pkg.from_code, pkg.to_code) in [('en', 'hi'), ('en', 'bn')]:
        print(f"Downloading: {pkg.from_code} → {pkg.to_code}")
        downloaded_path = pkg.download()  # <-- fix here
        package.install_from_path(downloaded_path)

# Step 4: Confirm installed languages
installed_languages = translate.get_installed_languages()
print("Installed languages:")
for lang in installed_languages:
    print(f"- {lang.code} : {lang.name}")
