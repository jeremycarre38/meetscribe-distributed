# CLAUDE.md — Tâche : Réécriture du module de capture audio synchronisé

## Contexte du projet

Voir le `CLAUDE.md` principal pour l'architecture complète.

Ce fichier décrit **uniquement** la tâche de réécriture du module de capture audio.

---

## Problème à résoudre

`meetscribe-record` (package PyPI de l'auteur original) capture le micro et
l'audio système via deux processus ffmpeg séparés mergés avec `amerge`.

Le décalage entre les deux sources est **non-déterministe** :
- Parfois le système démarre 2s après le micro
- Parfois le micro démarre 0.6s après le système
- `use_wallclock_as_timestamps` + `amerge` produit des warnings
  `Non-monotonous DTS` et un mauvais alignement des canaux

**Résultat :** les voix se chevauchent dans le fichier WAV final, ce qui
dégrade la diarisation YOU/REMOTE.

---

## Solution cible

Remplacer la capture ffmpeg par un script Python `record.py` qui utilise
`sounddevice` pour capturer les deux sources audio avec **la même clock**,
garantissant une synchronisation parfaite.

### Principe

`sounddevice` permet d'ouvrir plusieurs streams audio en Python et de les
lire frame par frame de façon synchronisée. En utilisant un seul thread
de capture avec deux `InputStream`, on garantit que les deux canaux sont
alignés temporellement.

---

## Spécifications du script `record.py`

### Interface CLI

```bash
# Lancer un enregistrement
python record.py \
  --mic "alsa_input.usb-Logitech_G935_Gaming_Headset-00.mono-fallback" \
  --system "alsa_output.usb-Logitech_G935_Gaming_Headset-00.analog-stereo.monitor" \
  --output /tmp/meeting.wav

# Détecter automatiquement les sources
python record.py --auto

# Lister les sources disponibles
python record.py --list-devices
```

### Comportement attendu

- Capture micro (canal gauche) + système (canal droit) simultanément
- Sortie : fichier WAV stéréo 16kHz 16-bit PCM
- Arrêt propre sur Ctrl+C avec flush des buffers
- Affichage du niveau audio en temps réel (optionnel)
- Log du décalage mesuré entre les deux sources au démarrage

### Format de sortie

Identique à meetscribe-record :
```
Canal gauche (0) = micro (YOU)
Canal droit  (1) = système (REMOTE)
Sample rate      = 16000 Hz
Format           = pcm_s16le (WAV)
```

---

## Stack technique

### Dépendances à installer

```bash
pip install sounddevice soundfile numpy
```

### Approche recommandée

Utiliser deux `sounddevice.InputStream` démarrés dans le même thread avec
des callbacks synchronisés, ou utiliser `sounddevice.rec()` avec un device
virtuel si disponible.

Alternative : utiliser `pyaudio` avec `pa.open()` sur les deux sources avec
le même `frames_per_buffer` pour garantir l'alignement.

**Important :** tester avec `sounddevice` en premier car c'est plus simple.
Si la synchronisation n'est pas parfaite, basculer sur une approche avec
un seul appel ffmpeg utilisant le device PipeWire natif au lieu de PulseAudio.

### Approche ffmpeg alternative (si sounddevice ne suffit pas)

Utiliser le device `pipewire` au lieu de `pulse` dans ffmpeg — PipeWire
a une meilleure gestion de la synchronisation multi-sources :

```bash
ffmpeg \
  -f pipewire -i "mic_source" \
  -f pipewire -i "sys_source" \
  -filter_complex "..." \
  output.wav
```

---

## Intégration avec meetscribe-distributed

Une fois `record.py` fonctionnel, il remplace l'appel à `meetscribe-record`
dans le flow `meet run`. Le fork `meetscribe-distributed` sera modifié pour
appeler `record.py` au lieu de `meet_record.capture`.

Le fichier WAV produit doit être **identique en format** à celui produit par
meetscribe-record pour que le reste du pipeline (transcription, diarisation,
PDF) fonctionne sans modification.

---

## Tests à effectuer

1. **Test de synchronisation basique**
   - Lancer `record.py`
   - Parler + jouer de la musique simultanément
   - Extraire les deux canaux et vérifier l'alignement :
   ```bash
   ffplay -af "pan=mono|c0=c0" /tmp/meeting.wav  # micro seul
   ffplay -af "pan=mono|c0=c1" /tmp/meeting.wav  # système seul
   ```

2. **Test de décalage**
   - Clapper des mains devant le micro pendant qu'un son joue
   - Vérifier que le clap apparaît au même timestamp sur les deux canaux

3. **Test durée longue**
   - Enregistrer 10 minutes
   - Vérifier qu'il n'y a pas de dérive de synchronisation dans le temps

4. **Test intégration**
   - Lancer `meet transcribe` sur le WAV produit
   - Vérifier que YOU/REMOTE sont bien assignés

---

## Contexte machine

- **OS :** Ubuntu 22.04.5 LTS
- **Audio :** PipeWire (avec compatibilité PulseAudio)
- **Casque :** Logitech G935 USB
  - Micro : `alsa_input.usb-Logitech_G935_Gaming_Headset-00.mono-fallback`
  - Système : `alsa_output.usb-Logitech_G935_Gaming_Headset-00.analog-stereo.monitor`
- **ffmpeg :** 4.4.2

## Problème connu avec ffmpeg 4.4.2

`use_wallclock_as_timestamps` + `amerge` produit des timestamps
non-monotones. PipeWire device direct (`-f pipewire`) pourrait résoudre ça
mais nécessite ffmpeg 5+ ou une build avec support PipeWire activé.

Vérifier : `ffmpeg -devices 2>&1 | grep -i pipe`

---

## Fichiers concernés dans meetscribe-distributed

```
meetscribe-distributed/
├── record.py              ← À CRÉER (nouveau module de capture)
├── meet/
│   └── cli.py             ← À MODIFIER pour appeler record.py au lieu de meet_record
├── CLAUDE.md              ← Ce fichier
└── venv/
    └── lib/.../meet_record/  ← À REMPLACER par record.py
```
