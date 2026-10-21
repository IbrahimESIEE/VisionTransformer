# Transformer d'estimation de profondeur sur FPGA (Kria KV260)

Implémentation matérielle d'un **vision transformer d'estimation de profondeur** : le modèle est entraîné en Python (PyTorch) sur le jeu de données **NYU Depth V2**, puis traduit en circuit avec **Vitis HLS** (C++) et intégré dans un design **Vivado** pour s'exécuter sur la carte **Xilinx Kria KV260**.

Projet réalisé pendant mon stage à Mons (Belgique), de mai à août 2026.

## Résultats

| | |
| --- | --- |
| Carte | Kria KV260 (xck26-sfvc784-2LV-c) |
| Vitesse | ~25 images par seconde |
| Fréquence de la logique programmable | 85 MHz |
| Nombre de poids | 72 992 |
| Quantification | virgule fixe 16 bits (Q6.10) |
| Écart avec le calcul sur PC | de l'ordre d'une fraction de mètre |

Deux architectures ont été développées : une version **itérative**, qui tient dans les ressources de la carte, et une version **série**, fournie à titre de comparaison mais qui dépasse les ressources disponibles.

### Block design Vivado

![Block design Vivado](images/block_design.png)

L'accélérateur `vitis_top_0` est au centre. Ses trois accès mémoire sont regroupés par un SmartConnect, et le processeur du Zynq UltraScale+ leur ouvre l'accès à la mémoire DDR.

### Carte de profondeur obtenue sur la carte

![Résultat sur la carte](images/resultat_profondeur.png)

## Chaîne de conception

```
PyTorch (entraînement)  →  export des poids  →  C++ (Vitis HLS)  →  IP  →  Vivado (bitstream)  →  KV260 (PYNQ / Jupyter)
```

## Structure du dépôt

```
├── python/          scripts d'entraînement du modèle et d'export des poids
├── Cpp/             code C++ du modèle, synthétisé par Vitis HLS
├── ip/              IP déjà synthétisées (ip_iterative et ip_serie)
├── fichierCarte/
│   ├── IterativeDesign.bit / .hwh   bitstream déjà généré
│   ├── TransformerDesign.tcl        script du block design Vivado
│   └── iterative.ipynb              notebook Jupyter qui pilote la carte
└── requirements.txt dépendances Python
```

L'IP du dossier `ip` et le bitstream de `fichierCarte` sont déjà générés : vous pouvez sauter les étapes 4 et 5 si vous ne voulez pas refaire la synthèse.

**Outils :** Python, PyTorch, Vitis HLS 2024.1, Vivado 2024.1, PYNQ, Jupyter.

---

## Reproduire le projet

Comptez environ une demi-journée, la synthèse matérielle étant de loin l'étape la plus longue.

### 1. Préparer Python et le jeu de données

Créez un environnement virtuel et installez les dépendances :

```bash
python -m venv monEnv
monEnv\Scripts\activate
pip install -r requirements.txt
```

Versions utilisées : torch 2.13.0, torchvision 0.28.0, numpy 2.4.6, pillow 12.3.0, matplotlib 3.11.1.

Téléchargez ensuite le jeu de données **NYU Depth V2** depuis Kaggle :

```bash
pip install kaggle
```

Créez un compte sur kaggle.com, allez dans **Settings > API > Create New Token**, puis placez le fichier `kaggle.json` dans `C:\Users\<votre nom>\.kaggle\`. Lancez alors le téléchargement :

```bash
kaggle datasets download -d awsaf49/nyuv2-official-split-dataset -p data --unzip
```

Prévoyez plusieurs gigaoctets d'espace disque. Vous devez obtenir un dossier `train` et un dossier `test/official`, contenant chacun des paires `rgb_xxxxx.png` et `depth_xxxxx.png`. Si le tout se trouve dans un sous-dossier supplémentaire, remontez-le d'un niveau.

### 2. Entraîner le modèle et exporter les poids

Dans le dossier `python` :

| Script | Rôle | À vérifier avant de lancer |
| --- | --- | --- |
| `entrainement.py` | Entraîne le modèle et enregistre les poids dans `checkpoints/last.pth` | Adapter `DATA_ROOT` et `CHECKPOINT_DIR` en haut du fichier |
| `exportPoids.py` | Écrit les poids et les valeurs de référence dans `hls_export/` | `PATCH_SIZE` doit valoir 16, comme dans `entrainement.py` |

```bash
python entrainement.py
python exportPoids.py
```

`dataset.py` ne se lance pas seul : il lit les images et il est appelé automatiquement par l'entraînement.

> Si un fichier `train_index.json` vide se trouve dans le dossier `python`, supprimez-le avant l'entraînement. C'est un cache de la liste des images : s'il est vide, le programme croit que le jeu de données est vide et s'arrête.

### 3. Du Python au C++

Créez un dossier de projet et rangez-y les fichiers du dossier `Cpp` comme ci-dessous (cette organisation est obligatoire, les fichiers s'appellent entre eux par ces chemins) :

```
monProjet/
├── codes/        vitis_top.cpp, model_parts.cpp, softmax.cpp et tous les .h
├── tb/           tb_vitis_top.cpp
├── hls_export/   copié depuis le dossier python
└── params.h      (généré ci-dessous)
```

Copiez `generate_params_h.py` et `weight_q16.py` à la racine de `monProjet`, puis lancez :

```bash
python generate_params_h.py
python weight_q16.py
```

- `generate_params_h.py` écrit `params.h` à la racine et `config.h` dans `codes` (il écrase volontairement le `config.h` fourni, qui doit correspondre au modèle entraîné).
- `weight_q16.py` crée `weights_q610.bin`, le fichier de poids lu par la carte. Il doit afficher **72992** poids. Gardez ce fichier pour l'étape 6.

Deux réglages à faire à la main :
- dans `codes/arch.h`, `#define ARCH` doit valoir `ARCH_ITERATIF` ;
- dans `tb/tb_vitis_top.cpp`, `#define DATA_DIR` doit pointer vers votre dossier `hls_export`.

### 4. Synthèse avec Vitis HLS 2024.1

Créez un composant HLS visant la KV260 (`xck26-sfvc784-2LV-c`) avec une période d'horloge de 10 ns :

| Emplacement dans Vitis HLS | Fichiers |
| --- | --- |
| Sources | `vitis_top.cpp`, `model_parts.cpp`, `softmax.cpp` |
| Test Bench | `tb_vitis_top.cpp` |
| Top Function | `vitis_top` |

Lancez dans l'ordre **C Simulation**, **C Synthesis** puis **Package**. Prévoyez plusieurs heures.

### 5. Bitstream avec Vivado 2024.1

Créez un projet RTL vide pour la carte Kria KV260, ouvrez la **Tcl Console** et tapez (en adaptant les chemins) :

```tcl
set_property ip_repo_paths "C:/.../ip/ip_iterative" [current_project]
update_ip_catalog -rebuild
source "C:/.../fichierCarte/TransformerDesign.tcl"
```

> Évitez les accents dans les chemins : Vivado ne les supporte pas.

Réglez l'horloge de la logique programmable sur 85 MHz :

```tcl
set_property CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ {85} [get_bd_cells zynq_ultra_ps_e_0]
validate_bd_design
```

Faites un clic droit sur le fichier `.bd` → **Create HDL Wrapper**, puis lancez **Generate Bitstream** (environ une heure). Récupérez ensuite :
- le bitstream : `.runs/impl_1/TransformerDesign_wrapper.bit`
- la description matérielle : `.gen/sources_1/bd/TransformerDesign/hw_handoff/TransformerDesign.hwh`

Les deux fichiers doivent avoir **le même nom de base**, par exemple `IterativeDesign.bit` et `IterativeDesign.hwh`, sinon la carte refusera de charger le design.

### 6. Exécution sur la carte

Connectez-vous à la carte en SSH :

```bash
ssh ubuntu@<IP_de_la_carte>
```

Créez un dossier de travail et démarrez Jupyter :

```bash
mkdir transformer
cd transformer
sudo -E env PATH=$PATH jupyter lab --no-browser --port=9090 --ip=0.0.0.0 --NotebookApp.token='' --NotebookApp.password='' --allow-root
```

Ouvrez `http://<IP_de_la_carte>:9090/lab` dans votre navigateur. Si le port 9090 est déjà utilisé (`port is already in use`), utilisez 9091 dans la commande **et** dans l'adresse.

Copiez dans ce dossier :

| Fichier | Provenance |
| --- | --- |
| `iterative.ipynb` | `fichierCarte` |
| `IterativeDesign.bit` / `.hwh` | `fichierCarte`, ou votre propre synthèse (étape 5) |
| `weights_q610.bin` | produit par `weight_q16.py` (étape 3) |
| `input_image.txt` / `final_depth_output.txt` | `hls_export/reference` |

Exécutez les cellules du notebook dans l'ordre : il charge le design sur la carte, envoie l'image et les poids en mémoire, lance l'accélérateur et affiche la carte de profondeur. Les dernières cellules mesurent la vitesse et comparent le résultat de la carte à celui calculé sur PC.

> Si le chargement du design échoue, la carte garde souvent en mémoire un ancien bitstream : redémarrez le noyau et relancez la première cellule.
