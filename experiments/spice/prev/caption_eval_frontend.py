import streamlit as st
import nltk
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
import networkx as nx
import matplotlib.pyplot as plt
import json # For displaying SPICE-like data

# --- Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(layout="wide")

# --- NLTK Resource Check/Download ---
@st.cache_resource # Caches the download process
def download_nltk_resources():
    """Downloads necessary NLTK resources if not already present."""
    resource_status = {
        "punkt": False,
        "punkt_tab": False,
        "wordnet": False,
        "omw-1.4": False,
        "averaged_perceptron_tagger": False
    }
    download_log = []

    # Helper to check and download
    def check_and_download(resource_name, check_path_part, download_id=None):
        if download_id is None:
            download_id = resource_name
        try:
            if resource_name == "wordnet_synsets_test": # Special check for wordnet functionality
                 wn.synsets("test")
            elif resource_name == "omw_lang_test": # Special check for omw functionality
                 wn.synsets("dog", lang='eng')
            elif resource_name == "pos_tag_test": # Special check for tagger functionality
                 nltk.pos_tag(["test"])
            else: # General path check
                 nltk.data.find(check_path_part)
            resource_status[resource_name if resource_name not in ["wordnet_synsets_test", "omw_lang_test", "pos_tag_test"] else download_id] = True
            download_log.append(f"Resource '{download_id}' already available.")
        except LookupError:
            download_log.append(f"Resource '{check_path_part}' for '{download_id}' not found. Downloading '{download_id}'...")
            try:
                nltk.download(download_id)
                resource_status[resource_name if resource_name not in ["wordnet_synsets_test", "omw_lang_test", "pos_tag_test"] else download_id] = True
                download_log.append(f"Successfully downloaded '{download_id}'.")
            except Exception as e:
                download_log.append(f"Error downloading '{download_id}': {e}")
        except Exception as e: # Catch other errors during check (e.g., if nltk itself is broken)
            download_log.append(f"Error checking resource '{download_id}': {e}")

    # Order matters somewhat (e.g., wordnet before omw-1.4)
    check_and_download("punkt", "tokenizers/punkt", "punkt")
    check_and_download("punkt_tab", "tokenizers/punkt_tab/english", "punkt_tab")
    check_and_download("wordnet_synsets_test", "", "wordnet")
    check_and_download("omw_lang_test", "", "omw-1.4")
    check_and_download("pos_tag_test", "taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger")

    # Final check for primary resources based on download_id names used in resource_status dict
    final_status = {
        "punkt": resource_status.get("punkt", False),
        "punkt_tab": resource_status.get("punkt_tab", False),
        "wordnet": resource_status.get("wordnet", False),
        "omw-1.4": resource_status.get("omw-1.4", False),
        "averaged_perceptron_tagger": resource_status.get("averaged_perceptron_tagger", False),
    }

    return {"status": final_status, "log": download_log}

# Call the download function once at the start after page config
nltk_download_info = download_nltk_resources()

# --- (1) Code from caption_comparator.py (with minor adjustments for Streamlit) ---
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos_for_lemmatizer(word):
    """Map POS tag to first character lemmatizer() accepts for better lemmatization."""
    # Use a simple mapping for common word endings
    if word.endswith('ed') or word.endswith('ing'):
        return wn.VERB
    elif word.endswith('ly'):
        return wn.ADV
    elif word.endswith('al') or word.endswith('ful') or word.endswith('ous'):
        return wn.ADJ
    else:
        return wn.NOUN  # Default to noun for simplicity

def get_synonyms_for_word(word):
    """Get a set of synonyms for a word using WordNet, including its lemmatized form."""
    synonyms = set()
    processed_word = word.lower() # Ensure word is lowercase
    lemma = lemmatizer.lemmatize(processed_word, get_wordnet_pos_for_lemmatizer(processed_word))
    for syn in wn.synsets(lemma, pos=get_wordnet_pos_for_lemmatizer(lemma)):
        for lem_obj in syn.lemmas():
            synonyms.add(lem_obj.name().lower().replace('_', ' '))
    synonyms.add(lemma)
    return synonyms

def compare_captions_binary_wordnet(candidate_caption, reference_caption):
    """
    Compares a candidate caption to a reference caption using binary matching
    with WordNet synonyms. Calculates precision, recall, and F1-score.
    """
    candidate_tokens = [token.lower() for token in word_tokenize(candidate_caption) if token.isalnum()]
    reference_tokens = [token.lower() for token in word_tokenize(reference_caption) if token.isalnum()]

    if not candidate_tokens or not reference_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0, "matched_words_in_candidate": []}

    matched_words_candidate = []
    ref_matched_flags = [False] * len(reference_tokens)

    for cand_token in candidate_tokens:
        cand_syns = get_synonyms_for_word(cand_token)
        for i, ref_token in enumerate(reference_tokens):
            if not ref_matched_flags[i]:
                ref_syns = get_synonyms_for_word(ref_token)
                if cand_syns.intersection(ref_syns):
                    matched_words_candidate.append(cand_token)
                    ref_matched_flags[i] = True
                    break

    num_matched_words = len(matched_words_candidate)
    precision = num_matched_words / len(candidate_tokens) if candidate_tokens else 0.0
    recall = num_matched_words / len(reference_tokens) if reference_tokens else 0.0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "matched_words_in_candidate": matched_words_candidate
    }

# --- (2) Code from spice_test_evaluator.py (Mocked SPICE) ---
def prepare_spice_input_data(gts_dict, res_dict):
    """Prepares data in the format SPICE typically expects."""
    input_data = []
    for img_id in gts_dict:
        if img_id in res_dict:
            input_data.append({
                "image_id": img_id,
                "test": res_dict[img_id][0],
                "refs": gts_dict[img_id]
            })
    return input_data

def run_mock_spice_evaluation(input_data_list):
    """
    Simulates SPICE evaluation, returning mock scores.
    This is a placeholder and does not reflect actual SPICE complexity.
    """
    mock_results = []
    if not input_data_list:
        return mock_results

    for item in input_data_list:
        cand_tokens = set(word_tokenize(item['test'].lower()))
        ref_overlap_scores = []
        for ref_text in item['refs']:
            ref_tokens = set(word_tokenize(ref_text.lower()))
            overlap = len(cand_tokens.intersection(ref_tokens)) / (len(cand_tokens.union(ref_tokens)) or 1)
            ref_overlap_scores.append(overlap)

        avg_overlap = sum(ref_overlap_scores) / len(ref_overlap_scores) if ref_overlap_scores else 0

        mock_f = min(0.9, avg_overlap * 1.5)
        mock_p = min(0.9, mock_f + 0.05)
        mock_r = min(0.9, mock_f - 0.05 if mock_f > 0.1 else 0.05)

        mock_results.append({
            "image_id": item["image_id"],
            "scores": {
                "All": {"f": round(mock_f, 3), "p": round(mock_p, 3), "r": round(mock_r, 3)},
                "Object": {"f": round(mock_f * 0.9, 3), "p": round(mock_p*0.9, 3), "r": round(mock_r*0.9, 3)},
                "Attribute": {"f": round(mock_f * 0.8, 3), "p": round(mock_p*0.8, 3), "r": round(mock_r*0.8, 3)},
                "Relation": {"f": round(mock_f * 0.85, 3), "p": round(mock_p*0.85, 3), "r": round(mock_r*0.85, 3)}
            }
        })
    return mock_results

# --- (3) Simplified Scene Graph Tuple Extractor & Visualizer ---
def get_pos_tag(word):
    """Enhanced rule-based POS tagger with more comprehensive rules."""
    word = word.lower()

    # Common verbs
    common_verbs = {'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had',
                   'do', 'does', 'did', 'run', 'runs', 'running', 'ran', 'sprint',
                   'sprints', 'sprinting', 'sprinted', 'move', 'moves', 'moving',
                   'moved', 'walk', 'walks', 'walking', 'walked'}

    # Common adjectives
    common_adjectives = {'big', 'small', 'large', 'tiny', 'dark', 'light', 'bright',
                        'green', 'blue', 'red', 'yellow', 'brown', 'black', 'white',
                        'grassy', 'beautiful', 'ugly', 'happy', 'sad', 'quick', 'slow'}

    # Common adverbs
    common_adverbs = {'quickly', 'slowly', 'happily', 'sadly', 'very', 'really',
                     'quite', 'rather', 'too', 'so', 'well', 'badly'}

    # Check common words first
    if word in common_verbs:
        return 'VB'
    elif word in common_adjectives:
        return 'JJ'
    elif word in common_adverbs:
        return 'RB'

    # Check word endings
    if word.endswith(('ed', 'ing', 'ize', 'ise', 'ify', 'ate')):
        return 'VB'
    elif word.endswith('ly'):
        return 'RB'
    elif word.endswith(('al', 'ful', 'ous', 'ive', 'able', 'ible', 'ic', 'ical', 'y')):
        return 'JJ'
    elif word.endswith(('s', 'es')):
        return 'NNS'
    elif word.endswith(('tion', 'sion', 'ment', 'ness', 'ity', 'ance', 'ence')):
        return 'NN'
    else:
        return 'NN'  # Default to noun

def simple_tuple_extractor(caption):
    """
    Enhanced rule-based tuple extractor for demonstration.
    This is not a robust scene graph parser like SPICE uses.
    """
    tuples = []
    sentences = sent_tokenize(caption)
    for sentence in sentences:
        tokens = word_tokenize(sentence)
        tagged_tokens = [(token, get_pos_tag(token)) for token in tokens]

        # Extract adjective-noun pairs
        for i in range(len(tagged_tokens) - 1):
            if tagged_tokens[i][1].startswith('NN') and tagged_tokens[i+1][1].startswith('JJ'):
                tuples.append((tagged_tokens[i][0], tagged_tokens[i+1][0]))
            elif tagged_tokens[i][1].startswith('JJ') and tagged_tokens[i+1][1].startswith('NN'):
                tuples.append((tagged_tokens[i+1][0], tagged_tokens[i][0]))

        # Extract subject-verb-object patterns
        nouns = [token for token, pos in tagged_tokens if pos.startswith('NN')]
        verbs = [token for token, pos in tagged_tokens if pos.startswith('VB')]

        if len(nouns) >= 1 and len(verbs) >= 1:
            subject = nouns[0]
            verb_phrase = " ".join(verbs)
            if len(nouns) > 1:
                object_ = nouns[1]
                tuples.append((subject, verb_phrase, object_))
            else:
                tuples.append((subject, verb_phrase))

        # Extract prepositional phrases
        for i in range(len(tagged_tokens) - 2):
            if tagged_tokens[i][1].startswith('NN') and tagged_tokens[i+1][0].lower() in {'in', 'on', 'at', 'through', 'over', 'under'}:
                if tagged_tokens[i+2][1].startswith('NN'):
                    tuples.append((tagged_tokens[i][0], tagged_tokens[i+1][0], tagged_tokens[i+2][0]))

    # Remove duplicates while preserving order
    seen_tuples = set()
    unique_tuples_final = []
    for tpl in tuples:
        frozen_representation = tuple(sorted(tpl)) if len(tpl) == 2 else tpl
        if frozen_representation not in seen_tuples:
            unique_tuples_final.append(tpl)
            seen_tuples.add(frozen_representation)
    return unique_tuples_final


def get_scene_graph_figure(tuples, caption_name="Scene Graph"):
    """Generates a Matplotlib figure visualizing the scene graph tuples."""
    graph = nx.MultiDiGraph()
    fig, ax = plt.subplots(figsize=(12, 8))

    if not tuples:
        ax.text(0.5, 0.5, "No tuples extracted for visualization.", ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig

    for tpl in tuples:
        if len(tpl) == 2:
            obj, attr_val = str(tpl[0]), str(tpl[1])
            graph.add_node(obj, type='object', label=obj)
            attr_node_label = f"{attr_val} (attr)"
            graph.add_node(attr_node_label, type='attribute', label=attr_val)
            graph.add_edge(obj, attr_node_label, label="is")
        elif len(tpl) == 3:
            subj, rel, obj = str(tpl[0]), str(tpl[1]), str(tpl[2])
            graph.add_node(subj, type='object', label=subj)
            graph.add_node(obj, type='object', label=obj)
            graph.add_edge(subj, obj, label=str(rel))
        elif len(tpl) == 1:
             graph.add_node(str(tpl[0]), type='object', label=str(tpl[0]))

    if not graph.nodes:
        ax.text(0.5, 0.5, "Graph has no nodes after processing tuples.", ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig

    pos = nx.spring_layout(graph, k=1.8, iterations=70, seed=42)

    node_colors = ["skyblue" if graph.nodes[n]['type'] == 'object' else "lightcoral" for n in graph.nodes()]
    node_labels = {n: graph.nodes[n]['label'] for n in graph.nodes()}

    nx.draw_networkx_nodes(graph, pos, ax=ax, node_size=3000, node_color=node_colors, alpha=0.9, linewidths=1, edgecolors='grey')
    nx.draw_networkx_edges(graph, pos, ax=ax, arrowstyle="-|>", arrowsize=25,
                           edge_color="darkgrey", width=2, alpha=0.8, connectionstyle="arc3,rad=0.1")
    nx.draw_networkx_labels(graph, pos, labels=node_labels, ax=ax, font_size=10, font_weight="bold")

    edge_labels_dict = nx.get_edge_attributes(graph, 'label')
    nx.draw_networkx_edge_labels(graph, pos, ax=ax, edge_labels=edge_labels_dict, font_size=9, font_color='firebrick', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=0.5))

    ax.set_title(caption_name, fontsize=16, fontweight="bold")
    plt.axis('off')
    return fig

# --- Streamlit App UI ---
st.title("📝 Caption Evaluation Dashboard")

# Display NLTK download logs in an expander for debugging/info
with st.expander("NLTK Resource Initialization Log"):
    st.write(nltk_download_info["log"])
    # st.write("Current NLTK Resource Status:")
    # st.json(nltk_download_info["status"])


st.markdown("""
This dashboard provides a simplified, interactive way to evaluate image captions.
- **Binary WordNet Comparison**: Compares captions based on word-to-word synonym matches.
- **Mock SPICE Evaluation**: Simulates SPICE input/output. _Actual SPICE requires a separate Java-based setup._
- **Conceptual Scene Graph**: Visualizes a *simplified, rule-based extraction* of scene elements. _This is NOT the output of SPICE's advanced parser._
""")

st.sidebar.header("Input Captions")
candidate_caption = st.sidebar.text_input("Candidate Caption:", "A large brown dog is running in the green field.")
reference_captions_text = st.sidebar.text_area("Reference Captions (one per line):",
                                         "A big, dark canine sprints through the grassy pasture.\nThe dog is moving quickly in a lawn.")

# Check if all essential NLTK resources were loaded successfully
all_resources_loaded = all(nltk_download_info["status"].values())
if not all_resources_loaded:
    missing = [res for res, status in nltk_download_info["status"].items() if not status]
    st.error(f"Critical NLTK resources could not be loaded: {', '.join(missing)}. Please check your internet connection and NLTK setup. The app may not function correctly.")
    st.stop()


if not candidate_caption:
    st.sidebar.warning("Please enter a candidate caption.")
    st.stop()

references = [ref.strip() for ref in reference_captions_text.split('\n') if ref.strip()]
if not references:
    st.sidebar.warning("Please enter at least one reference caption.")
    st.stop()

if st.sidebar.button("Evaluate Captions", type="primary", use_container_width=True):
    st.header("📊 Evaluation Results")

    st.subheader("1. Binary WordNet Synonym Comparison")
    if len(references) > 1:
        st.write("Comparing candidate with the **first** reference caption for this binary WordNet matching:")

    binary_results = compare_captions_binary_wordnet(candidate_caption, references[0])
    col1, col2, col3 = st.columns(3)
    col1.metric("Precision", f"{binary_results['precision']:.3f}")
    col2.metric("Recall", f"{binary_results['recall']:.3f}")
    col3.metric("F1-Score", f"{binary_results['f1_score']:.3f}")
    with st.expander("See matched words in candidate (vs. first reference)"):
        st.write(binary_results['matched_words_in_candidate'] or "No words matched.")

    st.subheader("2. Mock SPICE Evaluation (Illustrative)")
    st.info("Scores below are from a MOCK SPICE evaluator. Actual SPICE requires a separate, complex setup.")

    img_id = "eval_img_1"
    gts_data = {img_id: references}
    res_data = {img_id: [candidate_caption]}

    spice_input = prepare_spice_input_data(gts_data, res_data)
    mock_spice_output = run_mock_spice_evaluation(spice_input)

    if mock_spice_output and mock_spice_output[0]['scores']:
        spice_scores_all = mock_spice_output[0]['scores']['All']
        col_s1, col_s2, col_s3 = st.columns(3)
        col_s1.metric("Mock SPICE (F1)", f"{spice_scores_all['f']:.3f}")
        col_s2.metric("Mock Precision", f"{spice_scores_all['p']:.3f}")
        col_s3.metric("Mock Recall", f"{spice_scores_all['r']:.3f}")

        with st.expander("Detailed Mock SPICE Scores & Formatted Input"):
            st.write("**Formatted Input for SPICE (Illustrative):**")
            st.json(spice_input)
            st.write("**Mock SPICE Output (Illustrative):**")
            st.json(mock_spice_output)
    else:
        st.warning("Could not generate mock SPICE scores for the provided captions.")

    st.subheader("3. Conceptual Scene Graph Visualization (Simplified Parser)")
    st.info("The graphs below are generated from a *very basic rule-based parser* for illustrative purposes.")

    tab_titles = ["Candidate Caption Graph"]
    if references:
        tab_titles.append(f"Reference: \"{references[0][:30]}...\" Graph")

    tabs = st.tabs(tab_titles)

    with tabs[0]:
        st.markdown(f"**For Candidate:** \"{candidate_caption}\"")
        candidate_tuples = simple_tuple_extractor(candidate_caption)
        if candidate_tuples:
            st.write(f"*Extracted Tuples (Simplified):* `{candidate_tuples}`")
            fig_cand = get_scene_graph_figure(candidate_tuples, caption_name=f"Conceptual Graph: Candidate")
            st.pyplot(fig_cand)
        else:
            st.write("Could not extract sufficient tuples for candidate caption visualization.")

    if len(tabs) > 1:
        with tabs[1]:
            st.markdown(f"**For Reference 1:** \"{references[0]}\"")
            ref_tuples = simple_tuple_extractor(references[0])
            if ref_tuples:
                st.write(f"*Extracted Tuples (Simplified):* `{ref_tuples}`")
                fig_ref = get_scene_graph_figure(ref_tuples, caption_name=f"Conceptual Graph: Reference 1")
                st.pyplot(fig_ref)
            else:
                st.write("Could not extract sufficient tuples for reference caption visualization.")
else:
    st.info("Enter captions in the sidebar and click 'Evaluate Captions' to see the results.")

st.sidebar.markdown("---")
st.sidebar.markdown("Developed by [Gayanukaa](https://gayanukaa.github.io)")
st.sidebar.markdown(f"Streamlit Version: {st.__version__}, NLTK Version: {nltk.__version__}")
