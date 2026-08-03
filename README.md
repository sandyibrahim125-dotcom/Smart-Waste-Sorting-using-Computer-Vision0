# ♻️ Smart Waste Sorting using Computer Vision
### تصنيف المخلفات الذكي باستخدام الرؤية الحاسوبية

## 1. Project Description
This project classifies a photo of a waste item into one of six categories —
**cardboard, glass, metal, paper, plastic, trash** — using a MobileNetV2-based CNN
(transfer learning), and then recommends the correct recycling / treatment method
for that material. The goal is to make correct sorting easier at the point of
disposal (a camera on a bin, a phone app, a kiosk), which improves downstream
recycling yield at materials-recovery facilities.

**Repository contents**
- `notebook/Smart_Waste_Sorting_CV.ipynb` — full experimentation pipeline: EDA,
  preprocessing, baseline model, transfer learning, hyperparameter search,
  evaluation, and model export.
- `app.py` + `model/` — the Streamlit inference application.

## 2. Demo
- Deployed app: _add your Streamlit Cloud URL here once deployed_
- Notebook (view-only): open `notebook/Smart_Waste_Sorting_CV.ipynb` directly on GitHub,
  or run it in Colab/Jupyter locally.

## 3. Dataset
**TrashNet** (Gary Thung & Mindy Yang, 2016) — ~2,527 images, 6 classes
(`cardboard, glass, metal, paper, plastic, trash`).
- Repo: https://github.com/garythung/trashnet
- Kaggle mirror (alternative): search "Garbage Classification" dataset on Kaggle.

Expected local layout (used by the notebook and referenced by `DATA_DIR`):
```
data/dataset-resized/
  cardboard/
  glass/
  metal/
  paper/
  plastic/
  trash/
```

## 4. How to Run Locally

### 4.1 Clone and install
```bash
git clone <your-repo-url>
cd smart_waste_sorting
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 4.2 Train / export the model
Open `notebook/Smart_Waste_Sorting_CV.ipynb` in Jupyter (or Colab), point `DATA_DIR`
at your downloaded dataset, and run all cells. This produces:
```
exported_model/waste_classifier.keras
exported_model/class_indices.json
```
Copy both files into `model/artifacts/`:
```bash
cp exported_model/waste_classifier.keras model/artifacts/
cp exported_model/class_indices.json model/artifacts/
```

### 4.3 Run the app
```bash
streamlit run app.py
```
Then open the local URL Streamlit prints (usually http://localhost:8501).

## 5. Deployment on Streamlit Cloud (bonus)
1. Push this repo to GitHub (public).
2. Go to https://share.streamlit.io, connect your GitHub account.
3. Select this repository, branch, and set **`app.py`** as the entry point.
4. Make sure `model/artifacts/waste_classifier.keras` and `class_indices.json`
   are committed to the repo (or fetched at startup from a Release asset if too large).
5. Deploy — you'll get a public `https://<app-name>.streamlit.app` URL.

## 6. Project Structure
```
smart_waste_sorting/
├── app.py
├── model/
│   ├── __init__.py
│   ├── predict.py
│   ├── preprocess.py
│   └── artifacts/
│       ├── waste_classifier.keras   # exported from the notebook
│       └── class_indices.json       # exported from the notebook
├── notebook/
│   └── Smart_Waste_Sorting_CV.ipynb
├── requirements.txt
├── README.md
└── .gitignore
```

## 7. Recycling / Treatment Reference
| Class | Treatment / التدوير |
|---|---|
| Cardboard | Re-pulped into new paper products / يُحول للب الورق لصناعة منتجات ورقية |
| Glass | Crushed (cullet) and re-melted / يُكسر ويُصهر من جديد |
| Metal | Sorted magnetically/eddy-current, then melted into alloys / يُفرز ويُصهر |
| Paper | Pulped and reformed into new sheets / يُنقع ويُحول للب لصنع ورق جديد |
| Plastic | Sorted by resin type, shredded, reprocessed into pellets / يُفرز ويُجرش |
| Trash | Landfill or waste-to-energy / مكب صحي أو تحويل لطاقة |

## 8. Team Contribution
| Name | Role | Contribution |
|---|---|---|
| _Member 1_ | _Data & EDA_ | _..._ |
| _Member 2_ | _Modeling_ | _..._ |
| _Member 3_ | _Streamlit App_ | _..._ |
| _Member 4_ | _Deployment & README_ | _..._ |

_(Fill in with your actual team members before submitting.)_
