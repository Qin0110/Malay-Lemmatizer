import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.add_dll_directory(r"C:\vcpkg\installed\x64-windows\bin")

import logging
logging.basicConfig(level=logging.INFO)

import malaya

string = 'boleh dtg 8pagi esok tak atau minggu depan? 2 oktober 2019 2pm, tlong bayar rm 3.2k sekali tau'
string1 = 'xjdi ke, y u xsuke makan HUSEIN kt situ tmpt, i hate it. pelikle, pada'
string2 = 'i mmg2 xske mknn HUSEIN kampng tmpat, i love them. pelikle saye'
string3 = 'perdana menteri ke11 sgt suka makn ayam, harganya cuma rm15.50'
string4 = 'pada 10/4, kementerian mengumumkan, 1/100'
string5 = 'Husein Zolkepli dapat tempat ke-12 lumba lari hari ni'
string6 = 'Husein Zolkepli (2011 - 2019) adalah ketua kampng di kedah sekolah King Edward ke-IV'
string7 = '2jam 30 minit aku tunggu kau, 60.1 kg kau ni, suhu harini 31.2c, aku dahaga minum 600ml'
string8 = 'awak sangat hot ye 🔥🔥. 🔥🙂'
string9 = 'hanyalah rm2 ribu'
string10 = 'mulakn slh org boleh ,bila geng tuh kena slhkn jgk xboleh trima .. pelik , dia slhkn org bole hri2 crta sakau then bila kna bls balik xdpt jwb ,kata mcm biasa slh (parti sampah) 🤣🤣🤣 jgn mulakn dlu slhkn org kalau xboleh trima bila kna bls balik 🤣🤣🤣'
string11 = 'Pemimpin yg hebat, panahan2 fitnah tu akan dituju kepadanya.. harap DS terus bersabar. Jasa baik DS menjadi asbab di sana kelak mahupun rakyat yg terhutang budi juga..'
string12 = 'berehatlh najib.. sudah2 lh tu.. jgn buat rakyat hilang kepercyaan tu pda system kehakiman negara.. klu btl x slh kenapa x dibuktikan semasa sblm rayuan.. sudah lah tu kami dh letih dengan drama korang. ok'
string13 = 'DSNR satu satunya legasi kpd negara penyambung perjuangan bangsa melayu..jatuhnya beliau dek kerana fitnah dan dengkinya manusia..semoga Allah lindungi Najib Bin Razak dunia dan akhirat..Aamiin'
string14 = 'Muhammad Najib sbb malaysiakini dah daftar.... Klu dia fitnah...tertuduh boleh saman.... Klu berita2 yg x daftar...tu yg susah nak saman...sbb x tahu owner'

lm = malaya.language_model.kenlm(model = 'bahasa-wiki-news')
corrector = malaya.spelling_correction.probability.load(language_model = lm)
stemmer = malaya.stem.huggingface()
normalizer = malaya.normalizer.rules.load(corrector, stemmer)

after = normalizer.normalize(string)['normalize']

print('-' * 50)
print('Output:\n')
print(after)