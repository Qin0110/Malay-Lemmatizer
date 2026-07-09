run main to debug

streamlit leave it dulu

# Environment setup: Python 3.10.11 @ 3.10++
1. Download python 3.10++ first
2. `py -3.10 -m venv env_310`
3. `./env_310/Scripts/Activate`, `python --version` to check version

# If want to ENABLE GPU BOOST (Current no use)
# Download cuda toolkit 13.2 at NVIDIA archive toolkit
`https://developer.nvidia.com/cuda-13-2-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local`

4. `pip install -r requirements.txt`

# IF: pypi_kenlm failed to install (Linux environment based library based on C++, Windows variable different naming convention by default)
1. OPEN A NEW TERMINAL, THEN:
```bash
cd C:\
git clone https://github.com/microsoft/vcpkg.git
cd C:\vcpkg
.\bootstrap-vcpkg.bat
.\vcpkg install zlib:x64-windows
```

2. INSTALL zLIB
```bash
C:\vcpkg\installed\x64-windows\include\zlib.h
```

3. OPEN A NEW TERMINAL:
```bash
cd "D:\Degree in Data Science\NLP\New_Malay_Lemmatizer"
.\env\Scripts\activate
$env:PATH="C:\vcpkg\installed\x64-windows\bin;$env:PATH"
Test-Path C:\vcpkg\installed\x64-windows\bin\zlib1.dll
```

url: `https://chatgpt.com/share/6a4ffd98-c4b4-83ec-b382-06859e2883a6` (TIRED)