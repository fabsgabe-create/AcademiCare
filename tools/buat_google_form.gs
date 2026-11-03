/**
 * Membuat Google Form kuesioner stres (14 butir, skala 0-3) + spreadsheet respons dengan skor & kelas otomatis.
 * Isi butir sama dengan aplikasi AcademiCare (stressfer/questionnaire.py).
 *
 * CARA PAKAI
 * 1. Buka https://script.google.com -> Proyek baru -> hapus isi default -> tempel seluruh kode ini.
 * 2. Pilih fungsi "buatFormKuesioner" di menu atas -> Jalankan -> beri izin (akun Google Anda).
 * 3. Buka Lihat -> Log eksekusi: di sana ada tautan form (untuk dibagikan) dan tautan spreadsheet hasil.
 *
 * CATATAN: teks butir adalah TEMPLATE parafrase subskala stres gaya DASS. Untuk penelitian, ganti dengan instrumen baku
 * yang berizin (ubah array BUTIR di bawah) sesuai ketentuan komite etik Anda.
 */
var BUTIR = [
  'Saya marah karena hal-hal yang sepele.',
  'Saya cenderung bereaksi berlebihan terhadap suatu situasi.',
  'Saya merasa menghabiskan banyak energi karena cemas.',
  'Saya merasa gelisah.',
  'Saya merasa sulit untuk bersantai.',
  'Saya tidak sabar terhadap hal yang menghalangi pekerjaan saya.',
  'Saya merasa mudah tersinggung.',
  'Saya sulit menenangkan diri setelah sesuatu mengganggu saya.',
  'Saya sulit mentoleransi gangguan terhadap hal yang sedang saya kerjakan.',
  'Saya berada dalam keadaan tegang.',
  'Saya mudah marah.',
  'Saya sulit untuk beristirahat/menurunkan ketegangan.',
  'Saya merasa gugup secara berlebihan.',
  'Saya merasa tidak sabar ketika harus menunggu.'
];

function buatFormKuesioner() {
  var form = FormApp.create('Kuesioner Tingkat Stres Mahasiswa');
  form.setDescription(
    'Kuesioner ini bagian dari penelitian tesis "Deteksi dan Klasifikasi Tingkat Stres Akademik Mahasiswa Berbasis Citra ' +
    'Ekspresi Wajah". Isi sesuai kondisi Anda SELAMA SATU MINGGU TERAKHIR. Tidak ada jawaban benar atau salah. ' +
    'Hasil hanya untuk keperluan penelitian dan BUKAN diagnosis.\n\n' +
    'Keterangan skala:\n0 = Tidak sesuai dengan saya sama sekali\n1 = Sesuai dengan saya sampai tingkat tertentu / kadang-kadang\n' +
    '2 = Sesuai dengan saya sampai batas yang dapat dipertimbangkan / sering\n3 = Sangat sesuai dengan saya / sering sekali');
  form.setCollectEmail(false);
  form.setShowLinkToRespondAgain(false);
  form.setProgressBar(true);
  form.setConfirmationMessage(
    'Terima kasih atas partisipasi Anda. Jawaban sudah tersimpan. Bila Anda merasa tertekan, silakan hubungi psikolog atau ' +
    'layanan konseling kampus.');

  // Persetujuan (informed consent) - wajib
  var consent = form.addCheckboxItem();
  consent.setTitle('Persetujuan responden (informed consent)')
    .setHelpText('Data (NIM, nama, jawaban) dipakai peneliti hanya untuk penelitian ini dan dijaga kerahasiaannya. ' +
      'Partisipasi bersifat sukarela dan Anda boleh berhenti kapan saja.')
    .setChoices([consent.createChoice('Saya telah membaca dan menyetujui penggunaan data saya untuk penelitian ini')])
    .setRequired(true);

  form.addTextItem().setTitle('NIM').setHelpText('Gunakan NIM yang sama dengan akun aplikasi AcademiCare.').setRequired(true);
  form.addTextItem().setTitle('Nama lengkap').setRequired(true);

  // 14 butir dalam satu kisi: baris = pernyataan, kolom = skor 0-3
  var grid = form.addGridItem();
  grid.setTitle('Seberapa sesuai pernyataan berikut dengan kondisi Anda selama satu minggu terakhir?')
    .setHelpText('0 = tidak sesuai sama sekali | 1 = kadang-kadang | 2 = sering | 3 = sering sekali')
    .setRows(BUTIR)
    .setColumns(['0', '1', '2', '3'])
    .setRequired(true);

  // Spreadsheet respons + kolom skor dan kelas otomatis
  var ss = SpreadsheetApp.create('Hasil Kuesioner Tingkat Stres Mahasiswa');
  form.setDestination(FormApp.DestinationType.SPREADSHEET, ss.getId());
  var sheet = null;
  for (var i = 0; i < 10 && !sheet; i++) {            // sheet respons muncul beberapa detik setelah ditautkan
    Utilities.sleep(1500);
    sheet = ss.getSheets().filter(function (s) { return /Form Responses|Respons/i.test(s.getName()); })[0];
  }
  if (sheet) {
    var n = BUTIR.length;                              // kolom: A waktu, B persetujuan, C NIM, D nama, E..R butir
    var first = 'E', last = 'R';
    sheet.getRange('S1').setValue('Skor (0-' + (3 * n) + ')');
    sheet.getRange('T1').setValue('Kelas stres');
    sheet.getRange('S2').setFormula(
      '=ARRAYFORMULA(IF(A2:A="","",MMULT(IFERROR(VALUE(' + first + '2:' + last + '),0),SEQUENCE(' + n + ',1,1,0))))');
    sheet.getRange('T2').setFormula(
      '=ARRAYFORMULA(IF(S2:S="","",IF(S2:S<=14,"normal",IF(S2:S<=18,"ringan",IF(S2:S<=25,"sedang","berat")))))');
    sheet.getRange('S1:T1').setFontWeight('bold');
  } else {
    Logger.log('Sheet respons belum terbentuk. Setelah ada respons pertama, tambahkan kolom Skor dan Kelas secara manual.');
  }

  Logger.log('Tautan form (bagikan ke responden): ' + form.getPublishedUrl());
  Logger.log('Tautan edit form: ' + form.getEditUrl());
  Logger.log('Spreadsheet hasil: ' + ss.getUrl());
}
