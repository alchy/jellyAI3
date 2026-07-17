"""Testy kanonizace entit clusteringem variant."""

from jellyai.graph.canon import cluster_key, build_entity_canon


def test_case_variants_share_key():
    k = cluster_key("Božena Němcová")
    assert cluster_key("Boženy Němcové") == k
    assert cluster_key("Boženu Němcovou") == k


def test_dative_ovi_variants_share_key():
    """Dativ „-ovi" (Čapkovi) musí dát týž kmen jako ostatní pády (mezera stemmeru
    z handoffu: klíč (josf, čapkov) ≠ (josf, čapk) držel Čapkovi mimo cluster)."""
    assert cluster_key("Josefu Čapkovi") == cluster_key("Josef Čapek")
    assert cluster_key("Karlu Čapkovi") == cluster_key("Karel Čapek")


def test_distinct_people_differ():
    assert cluster_key("Karel Čapek") != cluster_key("Josef Čapek")
    assert cluster_key("Karel Čapek") != cluster_key("Karel Havlíček")


def test_canon_maps_variants_to_most_frequent():
    freq = {"Božena Němcová": 90, "Boženy Němcové": 41, "Boženu Němcovou": 8,
            "Karel Čapek": 53, "Josef Čapek": 19}
    canon = build_entity_canon(freq)
    # všechny Boženiny pády → nejčastější „Božena Němcová"
    assert canon["Boženy Němcové"] == "Božena Němcová"
    assert canon["Boženu Němcovou"] == "Božena Němcová"
    # různí lidé zůstanou oddělení
    assert canon["Karel Čapek"] == "Karel Čapek"
    assert canon["Josef Čapek"] == "Josef Čapek"


def test_places_case_merge():
    """Prostá pádová koncovka místa se sjednotí (souhlásková alternace Praze→
    „praz" je známý gap — h↔z lokativ, minorita)."""
    canon = build_entity_canon({"Praha": 49, "Prahy": 10})
    assert canon["Prahy"] == "Praha"


def test_deterministic_tie_break():
    """Při shodné četnosti je kanonické id v clusteru deterministické."""
    canon = build_entity_canon({"Praha": 1, "Prahy": 1})   # stejný cluster „prah"
    assert canon["Praha"] == canon["Prahy"]     # jeden reprezentant pro obě


def test_feminine_surname_and_vowel_length_merge():
    """Tři varianty téhož jména (miss-artefakt z Písma): ženské „-á" i délka
    samohlásky v kmeni se srovnají → jeden cluster."""
    k = cluster_key("Marie Magdalena Novotná")
    assert cluster_key("Marie Magdaleny Novotné") == k
    assert cluster_key("Marie Magdaléna Novotná") == k
