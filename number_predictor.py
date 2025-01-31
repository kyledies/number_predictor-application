###================================================================###
### Importing relevant modules
###================================================================###
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
import time

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import VotingClassifier

from streamlit_drawable_canvas import st_canvas
from PIL import Image
import cv2  # OpenCV används för bildmanipulation


###================================================================###
### Funktion för att importera MNIST-dataset och cache:ar detta i minnet.
@st.cache_data #Dekoratör som lagrar resultatet (X,y) i minnet 
def load_mnist():
    # Ladda MNIST dataset från OpenML
    mnist = fetch_openml('mnist_784', version=1, cache=True, as_frame=False)
    X = mnist["data"]
    y = mnist["target"].astype(np.uint8)
    print(X.shape)
    print(y.shape)
    return X, y

###================================================================###
### Skapar en dekoratör som cache:ar modellen - Den lagras in en gång och "används därefter ur minnet"
@st.cache_resource
def load_model():
    model_filename = "voting_classifier_model.joblib"  
    predictor_2000 = joblib.load(model_filename)
    return predictor_2000

###===========================TAB1=====================================###
### Funktion för testkörning av modellen på testdata
# Funktion för att göra prediktioner
def run_prediction():
    num_predictions = st.session_state.num_predictions

    # Starttid
    start_time = time.time()

    # Gör prediktioner
    X_test_subset = X_test[:num_predictions]  # Välj det antal testdata som användaren valde
    y_test_subset = y_test[:num_predictions]
    #st.write(X_test_subset)
    #st.write(X_test[1,:])
    y_pred = predictor_2000.predict(X_test_subset)


    case_accuracy = f"{100*predictor_2000.score(X_test_subset, y_test_subset):.2f}"  # Träffsäkerhet, formaterad till 2 decimaler

    # Beräkna och visa Confusion Matrix
    cm = confusion_matrix(y_test_subset, y_pred)

    end_time = time.time()
    elapsed_time = f"{end_time - start_time:.2f}"  # Beräknad tid, formaterad till 2 decimaler

    # Plottar confusion matrix
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y), ax=ax)
    ax.set_title(f"Confusion Matrix för: {num_predictions} prediktioner. Träffsäkerhet: {case_accuracy}%, Beräkningstid: {elapsed_time}s")
    ax.set_xlabel("Predikterade Värden")
    ax.set_ylabel("Sanna Värden")

    # **Spara figuren i session_state**
    st.session_state["prediction_figure"] = fig

###===========================TAB1=====================================###
### Funktion för att dynamiskt plotta exempel på hur datat ser ut:
def run_examples():
    # Hämta antalet exempel från slidern
    num_examples = st.session_state.num_examples

# Skapa en subplot med antal siffror = rader och 0,1,...9 => 10 kolumner
    fig, axes = plt.subplots(num_examples, 10, figsize=(15, num_examples * 1.5))
    
    # Om antalet rader är 1, gör om axes till en 2D-array för att undvika indexeringsfel
    if num_examples == 1:
        axes = np.expand_dims(axes, axis=0)

    # Skapa en lista över de etiketter vi vill visa (0, 1, 2,..., 9)
    for i in range(10):
        # Välj slumpmässiga index för varje siffra
        digit_indices = np.where(y == i)[0]
        random_digit_indices = np.random.choice(digit_indices, size=num_examples, replace=False)  # Välj slumpmässiga index
        
        # Plotta exempel för varje siffra
        for j, idx in enumerate(random_digit_indices):
            ax = axes[j, i]  # Välj rätt subplot (rad j, kolumn i)
            digit = X[idx]   # Hämta siffran
            digit_image = digit.reshape(28, 28)  # Omforma till en 28x28 bild
            ax.imshow(digit_image, cmap='binary')  # Visa bilden
            ax.set_title(f"True Label: {i}")  # Lägg till titel
            ax.axis('off')  # Ta bort axlar för renare plot
    plt.tight_layout()

    # **Spara figuren i session_state**
    st.session_state["example_figure"] = fig

###===========================TAB1=====================================###
### Funktion för att plotta slumpmässig siffra med specificerat målvärde:
def plot_digit(chosen_digit):
    # Hitta ett exempel för den valda siffran
    digit_indices = np.where(y == chosen_digit)[0]
    random_index = np.random.choice(digit_indices)  # Välj ett slumpmässigt index från den valda siffran
    some_digit_image = X[random_index].reshape(28, 28)

    # Skapa en figur och en axel för att visa bilden
    fig, ax = plt.subplots(figsize=(5, 5))

    # Visa bilden
    ax.imshow(some_digit_image, cmap='binary')

    # Lägg till titel baserat på den verkliga etiketten
    ax.set_title(f"True Label: {chosen_digit}")

    # Ta bort axlar för att göra bilden renare
    ax.axis('off')

    # Lägg till röd text för varje pixelvärde
    for i in range(28):
        for j in range(28):
            # Hämta värdet för pixeln
            pixel_value = some_digit_image[i, j]
            # Lägg till text för pixeln på rätt position
            ax.text(j, i, f'{pixel_value:.0f}', color='red', ha='center', va='center', fontsize=3.3, fontweight='bold')

    # Visa plotten i Streamlit
    st.pyplot(fig)

###===========================TAB2=====================================###
### Funktion för att centrera bild från canvas och bearbeta ritning till 28x28 (Chat gpt...)
# Funktion för att visa canvas och bearbeta ritningen till 28x28 pixlar
def center_image(img_array):
    """ Tar en 28x28 bild-array och centrerar genom att beskära tomrum och flytta tecknet till mitten. """

    # Skapa en binär mask där pixlar > 10 räknas som del av siffran
    _, thresh = cv2.threshold(img_array, 10, 255, cv2.THRESH_BINARY)

    # Hitta rader/kolumner som innehåller något annat än 0
    row_sum = np.any(thresh > 0, axis=1)  # True där det finns något ritat i rader
    col_sum = np.any(thresh > 0, axis=0)  # True där det finns något ritat i kolumner

    # Om hela bilden är tom, returnera en oförändrad bild
    if not np.any(row_sum) or not np.any(col_sum):
        return img_array

    # Bestäm bounding box genom att leta efter första/sista True i rad_sum/col_sum
    y_min, y_max = np.where(row_sum)[0][[0, -1]]
    x_min, x_max = np.where(col_sum)[0][[0, -1]]

    # Skär ut området med tecknet
    cropped = img_array[y_min:y_max+1, x_min:x_max+1]

    # Skapa en ny tom 28x28-bild
    new_img = np.zeros((28, 28), dtype=np.uint8)

    # Beräkna var vi ska placera det beskurna tecknet
    h, w = cropped.shape
    x_offset = (28 - w) // 2
    y_offset = (28 - h) // 2

    # Placera den beskurna bilden i mitten av den nya 28x28-bilden
    new_img[y_offset:y_offset+h, x_offset:x_offset+w] = cropped

    return new_img

###===========================TAB2=====================================###
### Funktion för att kunna rita på "canvas" i Streamlit
# Funktion för att visa canvas och bearbeta ritningen till 28x28 pixlar
def drawing_canvas():
    # Skapa två kolumner, en för canvas och en för den 28x28 figuren
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Rita  vald siffra på Canvas:")
        # Välj siffra att skriva
        target_digit = st.selectbox("Välj vilken siffra du ska skriva:", list(range(10)))
        # Skapa canvas
        canvas_result = st_canvas(
            fill_color="white",  # Bakgrundsfärg
            stroke_color="black",  # Ritfärg
            stroke_width=30,  # Ritpenselns tjocklek
            background_color="white",  # Bakgrundsfärg på canvas
            width=400,  # Canvas bredd
            height=400,  # Canvas höjd
            drawing_mode="freedraw",  # Rita i frihand
            key="canvas",
            display_toolbar=True,  # Visa verktygsfält för att rensa eller ändra ritverktyg
        )

    with col2:
        st.subheader("Förhandsgranskning av ritningen (28x28)")
        # Spara statistik i session_state för att uppdatera noggrannhet
        if 'total_attempts' not in st.session_state:
            st.session_state.total_attempts = 0
        if 'correct_attempts' not in st.session_state:
            st.session_state.correct_attempts = 0
        if 'accuracy_history' not in st.session_state:
            st.session_state.accuracy_history = []  # Lista för att lagra träffsäkerheten över tid

        # Knapp för att nollställa statistik
        if st.button("Återställ statistik!"):
            st.session_state.total_attempts = 0
            st.session_state.correct_attempts = 0
            st.session_state.accuracy_history = []

        if canvas_result.image_data is not None:
            
            ######################################
            ### HÄR BEARBETAS INPUT FRÅN CANVAS###
            ######################################

            # Omvandla ritning till PIL-bild och konvertera till gråskala (mha python-bibl) 
            pil_image_resized = Image.fromarray(canvas_result.image_data).convert('L').resize((28, 28), Image.Resampling.LANCZOS)

            ### Förbehandla bilden för modellens prediktion
            img_array = np.array(pil_image_resized)  # Omvandla till en numpy-array
            contrast_solver = 4
            if (contrast_solver==1):
                # Förstärk kontrasten på något vis
                img_array = cv2.equalizeHist(img_array)
            elif(contrast_solver==2):
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                img_array = clahe.apply(img_array)
            elif(contrast_solver==3):
                clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
                img_array = clahe.apply(img_array)
            elif(contrast_solver==4):
                clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(3, 3))
                img_array = clahe.apply(img_array)

            # Invertera (MNIST har svart text på vit bakgrund)
            img_array = 255 - img_array 

            # Normalisera och skala upp (för bättre kontrast)
            max_pixel_value = np.max(img_array)
            if max_pixel_value > 0:
                img_array = (img_array / max_pixel_value) * 255
                img_array = img_array.astype(np.uint8)  

            # Centrera bilden innan vi plattar ut den (funktion hämtad)
            img_array = center_image(img_array)

            # Platta ut till (1, 784) för att matcha modellens input
            img_array = img_array.reshape(1, 784)

            # Visa den centrerade bilden för debugging
            st.image(img_array.reshape(28, 28), caption="Centrerad ritning", clamp=True, use_container_width=True)

            # Hitta det högsta värdet i bilden och printar för check
            max_pixel_value = np.max(img_array)
            st.write("Maxpix", max_pixel_value)
            
            # Gör en prediktion
            prediction = predictor_2000.predict(img_array)
            predicted_digit = int(prediction[0])

            # När bild-array finns (ej tom canvas) 
            if np.max(img_array) != 0:
                # Uppdatera statistik
                st.session_state.total_attempts += 1
                if predicted_digit == target_digit:
                    st.session_state.correct_attempts += 1

                # Visa resultat, beräkna och visa träffsäkerheten samt uppdatera lista med resultat
                accuracy = (st.session_state.correct_attempts / st.session_state.total_attempts) * 100
                st.session_state.accuracy_history.append(accuracy)
                st.subheader(f"Modellens gissning: {prediction[0]}")
                st.write(f"**Träffsäkerhet:** {st.session_state.correct_attempts} av {st.session_state.total_attempts}	:arrow_right:  {accuracy:.2f}%")

    # Plotta statistikdiagram om vi har mer än 0 försök
    #if st.session_state.total_attempts > 0 and len(st.session_state.accuracy_history) == st.session_state.total_attempts:
    if st.session_state.total_attempts > 0 and len(st.session_state.accuracy_history) > 0:

        st.subheader("Träffsäkerhet för denna session:")
        # Plotta statistikdiagram
        fig, ax = plt.subplots(figsize=(6, 1.5))  # Anpassa figurens storlek
        ax.plot(range(1, st.session_state.total_attempts + 1), st.session_state.accuracy_history, marker='o', color='b')
        ax.set_xlabel("Antal Försök")
        ax.set_ylabel("Träffsäkerhet (%)")
        ax.set_title("Träffsäkerhet över försök")
        st.pyplot(fig)
    else:
        st.write("Ingen statistik att visa än!")

    st.subheader("Heatmaps och matris för vår input-data VS godtycklig MNIST-data:")
    col1, col2 = st.columns(2)

    with col1:
        # Skapa en figur och axlar om Canvas ej är tom
        if np.max(img_array) != 0:
            fig, ax = plt.subplots(figsize=(4, 4))  # Anpassa storleken

            # Rita heatmap
            sns.heatmap(img_array.reshape(28, 28), cmap="viridis", ax=ax, cbar=False, square=True, xticklabels=False, yticklabels=False)

            # Visa i Streamlit
            st.pyplot(fig)
            st.write("Indata i 28x28-form:", img_array.reshape(28,28))
        else:
            st.write("Ingen heatmap eftersom ingen siffra har ritats.")
    with col2:
        digit_ind = np.where(y == target_digit)[0] #Index för mnist där siffra matchar val
        random_idx = np.random.choice(digit_ind) #sSlumpar en siffra
        # Skapa en figur och axlar om Canvas inte är tom
        # Kontrollera om img_array är full med nollor
        if np.max(img_array) != 0:
            # Skapa en figur och axlar
            fig, ax = plt.subplots(figsize=(4, 4))  # Anpassa storleken

            # Rita heatmap
            sns.heatmap(X[random_idx].reshape(28, 28), cmap="viridis", ax=ax, cbar=False, square=True, xticklabels=False, yticklabels=False)

            # Visa i Streamlit
            st.pyplot(fig)
            st.write("Mnist indata i 28x28-form:", X[random_idx].reshape(28, 28), "Målvärde:", y[random_idx])
        else:
            st.write("Ingen heatmap eftersom ingen siffra har ritats.")

###############################################################################################
###############################################################################################
############################## Här startar vår applikation ####################################
###############################################################################################
###############################################################################################
###############################################################################################

# Ladda data
X, y = load_mnist()
# Dela upp data i tränings- och testdata
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=10000, random_state=42, stratify=y)
# Laddar in vår modell
predictor_2000 = load_model()  # Ladda modellen


st.title("Sifferprediktioner med ensemble learning och databearbetning")

# Huvudlogik med flikar
tabs = st.radio("Välj flik", ["Start", "Egna Prediktioner", "Om appen"])  # Använd radio för att byta mellan flikar
#tabs = st.sidebar.selectbox("Välj flik", ["Start", "Egna Prediktioner", "Om appen"])

if tabs == "Start":
    st.header("Välkommen - Här får du en kort introduktion till denna applikation för sifferprediktioner med hjälp av AI!", divider=True)
    st.write("""
        Denna applikation använder en tränad AI-modell som byggts med hjälp av Ensemble Learning för att prediktera siffror.
        Modellen som används skapades ur följande approach:
    """)
    st.markdown("""
1. **9 st maskininlärningsalgoritmer** tränades på 8.000 siffror från MNIST-datat med optimerade hyperparametrar (via Gridsearch). Validering skedde mot 2.000 siffror.
2. **Algoritmer med träffsäkerhet över 90%** från steg 1 inkluderades i en Voting Classifier. Dessa tränades sedan på 50.000 siffror ($X_{train}$) och validerades mot 10.000 siffror ($X_{val}$).
3. **Den bästa modellen (Voting Classifier)** tränades om på hela tränings- och valideringsdatasetet (60.000 siffror), testades på testdatat ($X_{test}$, 10.000st siffror) och sparades för vidare användning.

**De nio algoritmerna inkl bästa hyperparametrar (Gröna modeller: Träffsäkerhet > 90% på testdata och ingår i Voting Classifier.):**
- :white_check_mark: <span style="color:green; font-weight:bold;">Logistisk Regression</span> ({'C': 0.01, '_max_iter': 100, 'solver': 'lbfgs'}, Inkl pipeline med standardscaler())
- :white_check_mark: <span style="color:green; font-weight:bold;">Linjär SVC</span> ({'C': 0.1, 'max_iter': 1000, 'tol': 0.0001}, Inkl pipeline med standardscaler())
- :white_check_mark: <span style="color:green; font-weight:bold;">SVC</span> ({'C': 5, 'gamma': 'auto', 'kernel': 'rbf'}, Inkl pipeline med standardscaler())
- :white_check_mark: <span style="color:green; font-weight:bold;">Random Forest</span> ({'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 200})
- :white_check_mark: <span style="color:green; font-weight:bold;">Extra Trees</span> (Optimala hyperparametrar ej sparade i text.)
- :white_check_mark: <span style="color:green; font-weight:bold;">K-nearest Neighbors</span> ({'metric': 'manhattan', 'n_neighbors': 3, 'weights': 'distance'}, Inkl pipeline med standardscaler())
- :waving_white_flag: Gaussian Naive Bayes (Optimala hyperparametrar ej sparade i text.)
- :white_check_mark: <span style="color:green; font-weight:bold;">MLP</span> ({'activation': 'relu', 'hidden_layer_sizes': (100,), 'learning_rate': 'constant', 'max_iter': 200, 'solver': 'adam'}, Inkl pipeline med standardscaler())
- :goat: Voting Classifier - Vår bästa modell som efter träning predikterade $97.74\\%$ rätt på träningsdatat $X_{test}$

*Tränade av misstag två stycken SVC och lät dessa ingå i Voting Classifier.
""", unsafe_allow_html=True)

    st.subheader("Visualiering av :blue[MNIST-datat] :eyes:", divider=True)
    st.markdown(""" 
        Datasetet MNIST består utav handskrivna siffror vars features lagras vår $X$-variabel, samt motsvarande etiketter i variabel $y$.
        $X$-variablerna består av 784st features, som vid användning av *reshape(28,28)* ger en grov illustration av siffran.
        Nedan kan du välja ett litet slumpmässigt utdrag ur datat för illustration:
    """)

    if 'num_examples' not in st.session_state:
        st.session_state.num_examples = 1  # Initialvärde för exempel-slider

    num_examples = st.slider(
        "Välj antalet exempel per siffra ur MNIST-dataset (1-5):",
        min_value=1,
        max_value=5,
        step=1,
        key="num_examples",
        on_change=run_examples#on_change=lambda: st.session_state.update({"example_figure": run_examples()})
        )

    # Kör `run_examples()` vid start om ingen figur finns
    if "example_figure" not in st.session_state:
        run_examples()

    # Visa figuren
    st.pyplot(st.session_state["example_figure"])

    st.markdown("Välj nedan en siffra för att visualisera ett exempel av denna ur datasetet - Dess numeriska representation är inkluderat i figuren:")

    # dropdown för att välja en siffra, key gör den oberoende
    chosen_digit = st.selectbox("Välj en siffra (0-9):", options=range(10), key="digit_selector")

    # Anropa funktionen för att plottar den valda siffran
    plot_digit(chosen_digit)

    st.markdown("""
**Problematiken som kommer i nästa flik där prediktioner ska göras på egen data kommer framförallt att beröra punkter som:**

- Hur ska vi lyckas bearbeta egen data så att dess representation blir likvärdig den ovan?
    - Hur säkerställer vi centrering av data?
    - Hur ska datat skalas så att enbart siffrans pixlar får ett pixelvärde > 0?
- Med mera...
    """)

    st.subheader("Testa gärna modellen nedan på ett urval från $X_{test}$ :blue[MNIST-datat] :sunglasses:", divider=True)

        # Slider för att välja antalet prediktioner
    if 'num_predictions' not in st.session_state:
        st.session_state.num_predictions = 10  # Initialt värde om ingen har justerat slidern än

    num_predictions = st.slider(
        "Välj antalet prediktioner (mellan 10 och 1000):",
        min_value=1,
        max_value=1000,
        step=10,
        key="num_predictions",
        on_change=run_prediction 
    )

    # Kontrollera om en figur redan skapats och visa den
    if "prediction_figure" in st.session_state:
        st.pyplot(st.session_state["prediction_figure"])  # Visa figuren om den finns i session_state

elif tabs == "Egna Prediktioner":
    st.header("Prediktioner från Ritning", divider=True)
    st.subheader("Gör så här vid användning av modell nedan:")
    st.markdown("""
- **Välj siffra du ska rita**
- **Rita denna siffra (helst utan att släppa muspekaren från ritytan förrän du är klar)**
- **Vid byte av siffra eller återställning av statistik: Nollställ ritningen först!**
""")
    # Visa rit-canvasen här
    canvas_result = drawing_canvas()
    st.header("Prediktioner från Kamera", divider=True)


elif tabs == "Om Appen":
    st.header("", divider=True)
    st.write("fdjksf")


