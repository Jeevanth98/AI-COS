from __future__ import annotations

import json
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip().lower()


def token_set(value: Any) -> set[str]:
    text = normalize_text(value)
    if not text:
        return set()
    return set(re.findall(r"[a-z0-9]+", text))


def policy_type_key(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower().replace(" ", "_")


def same_or_overlap(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return False
    ta = normalize_text(a)
    tb = normalize_text(b)
    if ta == tb:
        return True
    return bool(token_set(ta) & token_set(tb))


def rule_similarity(rule_a: Dict[str, Any], rule_b: Dict[str, Any]) -> float:
    score = 0.0
    if same_or_overlap(rule_a.get("actor"), rule_b.get("actor")):
        score += 0.35
    if same_or_overlap(rule_a.get("action"), rule_b.get("action")):
        score += 0.45
    if same_or_overlap(rule_a.get("condition"), rule_b.get("condition")):
        score += 0.20
    if same_or_overlap(rule_a.get("temporal_constraint"), rule_b.get("temporal_constraint")):
        score += 0.20
    if same_or_overlap(rule_a.get("threshold"), rule_b.get("threshold")):
        score += 0.15
    if same_or_overlap(rule_a.get("document_id"), rule_b.get("document_id")):
        score += 0.10
    return score


class KnowledgeReasoningEngine:
    def __init__(self, dataset_root: str | Path):
        self.dataset_root = Path(dataset_root)
        self.processed_dir = self.dataset_root / "processed"
        self.registry = load_json(self.processed_dir / "document_registry.json")
        self.graph_data = load_json(self.processed_dir / "knowledge_graph.json")
        self.relationships = load_json(self.dataset_root / "relationships.json")
        self.rules = load_json(self.processed_dir / "structured_rules.json")
        self.scenarios = self._load_json_group(self.dataset_root / "scenarios")
        self.conflict_cases = self._load_json_group(self.dataset_root / "conflicts")
        self.document_rules = defaultdict(list)
        for rule in self.rules:
            self.document_rules[rule["document_id"]].append(rule)
        self.graph = self._build_graph()

    def _load_json_group(self, folder: Path) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        if not folder.exists():
            return items
        for file_path in sorted(folder.glob("*.json")):
            try:
                data = load_json(file_path)
                if isinstance(data, dict):
                    items.append(data)
            except Exception:
                continue
        return items

    def _build_graph(self) -> Dict[str, List[Tuple[str, str]]]:
        adjacency: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        for edge in self.graph_data.get("edges", []):
            if len(edge) >= 3:
                source, target, data = edge[0], edge[1], edge[2]
            elif len(edge) == 2:
                source, target = edge
                data = {"relationship": "REFERENCES"}
            else:
                continue
            relationship = str(data.get("relationship", "REFERENCES")).upper()
            adjacency[source].append((target, relationship))
            adjacency[target].append((source, relationship))
        return dict(adjacency)

    def _relation_weight(self, rel_type: str) -> float:
        weights = {
            "AFFECTS": 2.0,
            "CONTROLS": 1.7,
            "INFLUENCES": 1.3,
            "REFERENCES": 1.0,
        }
        return weights.get(str(rel_type).upper(), 0.8)

    def _neighbors_with_paths(self, start_doc: str, max_depth: int = 2) -> Dict[str, Dict[str, Any]]:
        visited: Dict[str, Dict[str, Any]] = {start_doc: {"depth": 0, "path": [start_doc], "reason": "changed_document"}}
        queue: deque[Tuple[str, int]] = deque([(start_doc, 0)])
        while queue:
            node, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for neighbor, relation in self.graph.get(node, []):
                if neighbor in visited:
                    continue
                visited[neighbor] = {
                    "depth": depth + 1,
                    "path": visited[node]["path"] + [neighbor],
                    "reason": f"{relation}",
                }
                queue.append((neighbor, depth + 1))
        return visited

    def _rule_overlap_score(self, doc_id: str, changed_rule: Dict[str, Any]) -> float:
        best = 0.0
        for rule in self.document_rules.get(doc_id, []):
            best = max(best, rule_similarity(changed_rule, rule))
        return best

    def analyze_sipa_scenario(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        changed_doc = scenario.get("changed_document")
        old_rule = scenario.get("old_rule", {})
        new_rule = scenario.get("new_rule", {})
        expected_affected = set(scenario.get("expected_affected_documents", []))
        impacted: Dict[str, Dict[str, Any]] = {}

        visited = self._neighbors_with_paths(changed_doc, max_depth=2)
        for doc_id, meta in visited.items():
            if doc_id == changed_doc:
                continue
            score = 0.0
            reasons: List[str] = []
            for neighbor, relationship in self.graph.get(doc_id, []):
                if neighbor == changed_doc:
                    score += self._relation_weight(relationship)
                    reasons.append(f"Direct relationship: {relationship} from {changed_doc}")
            for candidate_doc, candidate_meta in visited.items():
                if candidate_doc == doc_id:
                    continue
                for direct_neighbor, relationship in self.graph.get(candidate_doc, []):
                    if direct_neighbor == doc_id:
                        score += 0.5 * self._relation_weight(relationship)
                        reasons.append(f"Connected via {candidate_doc} using {relationship}")
            rule_overlap = self._rule_overlap_score(doc_id, new_rule)
            if rule_overlap > 0:
                score += 1.5 * rule_overlap
                reasons.append("Rule similarity suggests the same policy concept is relevant")
            if score > 0.0 or doc_id in expected_affected:
                impacted[doc_id] = {
                    "score": round(score, 3),
                    "reasons": reasons if reasons else ["Related policy topic or dependency path detected"],
                }

        sorted_impacted = [
            {
                "document_id": doc_id,
                "score": info["score"],
                "reasons": info["reasons"],
            }
            for doc_id, info in sorted(impacted.items(), key=lambda item: (-item[1]["score"], item[0]))
        ]

        evaluation = {
            "scenario_id": scenario.get("scenario_id"),
            "changed_document": changed_doc,
            "impacted_documents": [item["document_id"] for item in sorted_impacted],
            "reasons": {item["document_id"]: item["reasons"] for item in sorted_impacted},
        }
        return evaluation

    def evaluate_sipa(self) -> Dict[str, Any]:
        results = []
        for scenario in self.scenarios:
            predicted = self.analyze_sipa_scenario(scenario)
            expected = set(scenario.get("expected_affected_documents", []))
            predicted_set = set(predicted["impacted_documents"])
            tp = len(expected & predicted_set)
            fp = len(predicted_set - expected)
            fn = len(expected - predicted_set)
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
            results.append({
                "scenario_id": scenario.get("scenario_id"),
                "expected": sorted(expected),
                "predicted": sorted(predicted_set),
                "precision": round(precision, 3),
                "recall": round(recall, 3),
                "f1": round(f1, 3),
                "matches_expected": predicted_set == expected,
            })
        metrics = {
            "mean_precision": round(sum(item["precision"] for item in results) / len(results), 3) if results else 0.0,
            "mean_recall": round(sum(item["recall"] for item in results) / len(results), 3) if results else 0.0,
            "mean_f1": round(sum(item["f1"] for item in results) / len(results), 3) if results else 0.0,
            "scenario_results": results,
        }
        return metrics

    def detect_conflict(self, rule_a: Dict[str, Any], rule_b: Dict[str, Any]) -> bool:
        actor_a = normalize_text(rule_a.get("actor"))
        actor_b = normalize_text(rule_b.get("actor"))
        action_a = normalize_text(rule_a.get("action"))
        action_b = normalize_text(rule_b.get("action"))
        type_a = policy_type_key(rule_a.get("type") or rule_a.get("policy_type"))
        type_b = policy_type_key(rule_b.get("type") or rule_b.get("policy_type"))
        condition_a = normalize_text(rule_a.get("condition"))
        condition_b = normalize_text(rule_b.get("condition"))
        temporal_a = normalize_text(rule_a.get("temporal_scope") or rule_a.get("temporal_constraint"))
        temporal_b = normalize_text(rule_b.get("temporal_scope") or rule_b.get("temporal_constraint"))
        threshold_a = normalize_text(rule_a.get("limit") or rule_a.get("threshold"))
        threshold_b = normalize_text(rule_b.get("limit") or rule_b.get("threshold"))

        actor_match = bool(not actor_a or not actor_b or actor_a == actor_b or (actor_a and actor_b and token_set(actor_a) & token_set(actor_b)))
        action_overlap = bool(action_a and action_b and (action_a == action_b or token_set(action_a) & token_set(action_b)))
        condition_overlap = bool(condition_a and condition_b and (condition_a == condition_b or token_set(condition_a) & token_set(condition_b)))
        temporal_overlap = bool(not temporal_a or not temporal_b or temporal_a == temporal_b or token_set(temporal_a) & token_set(temporal_b))
        no_exception = True

        if not actor_match or not action_overlap:
            return False

        if {type_a, type_b} & {"permission", "obligation", "prohibition"}:
            if type_a != type_b and (temporal_overlap or condition_overlap) and no_exception:
                return True

        if threshold_a and threshold_b and threshold_a != threshold_b and (temporal_overlap or condition_overlap):
            return True

        if condition_overlap and temporal_overlap and type_a == type_b and action_overlap and actor_match:
            if threshold_a and threshold_b and threshold_a != threshold_b:
                return True

        return False

    def analyze_conflict_case(self, conflict_case: Dict[str, Any]) -> Dict[str, Any]:
        rule_a = conflict_case.get("rule_a", {})
        rule_b = conflict_case.get("rule_b", {})
        conflict_detected = self.detect_conflict(rule_a, rule_b)
        reason = "No direct contradiction identified"
        if conflict_detected:
            type_a = policy_type_key(rule_a.get("type") or rule_a.get("policy_type"))
            type_b = policy_type_key(rule_b.get("type") or rule_b.get("policy_type"))
            if type_a != type_b:
                reason = f"Policy types differ ({type_a} vs {type_b}) while actor and action overlap."
            else:
                reason = "Threshold or temporal conditions create a semantic conflict under the same actor-action scope."
        return {
            "conflict_id": conflict_case.get("conflict_id"),
            "documents": [conflict_case.get("document_a"), conflict_case.get("document_b")],
            "conflict_detected": conflict_detected,
            "reason": reason,
            "recommended_actions": [
                "Confirm the exact policy authority and precedence for both documents.",
                "Verify whether an exception or scope condition resolves the contradiction.",
                "Document the required implementation change for affected stakeholders."
            ]
        }

    def evaluate_conflicts(self) -> Dict[str, Any]:
        results = []
        for conflict_case in self.conflict_cases:
            predicted = self.analyze_conflict_case(conflict_case)
            expected_conflict = bool(conflict_case.get("expected_conflict"))
            matches = predicted["conflict_detected"] == expected_conflict
            results.append({
                "conflict_id": conflict_case.get("conflict_id"),
                "expected_conflict": expected_conflict,
                "predicted_conflict": predicted["conflict_detected"],
                "matches_expected": matches,
            })
        accuracy = round(sum(1 for item in results if item["matches_expected"]) / len(results), 3) if results else 0.0
        return {"accuracy": accuracy, "results": results}


def run_phase2(dataset_root: str | Path = "E:/SEM-07/NLP/Proj/Dataset/AI-COS_Dataset") -> Dict[str, Any]:
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    engine = KnowledgeReasoningEngine(root)
    sipa_results = []
    for scenario in engine.scenarios:
        sipa_results.append(engine.analyze_sipa_scenario(scenario))

    ascd_results = []
    for case in engine.conflict_cases:
        ascd_results.append(engine.analyze_conflict_case(case))

    sipa_evaluation = engine.evaluate_sipa()
    ascd_evaluation = engine.evaluate_conflicts()

    phase2_output = {
        "dataset_root": str(root),
        "sipa_results": sipa_results,
        "ascd_results": ascd_results,
        "sipa_evaluation": sipa_evaluation,
        "ascd_evaluation": ascd_evaluation,
    }

    output_path = root / "processed" / "phase2_reasoning_outputs.json"
    output_path.write_text(json.dumps(phase2_output, indent=2, ensure_ascii=False), encoding="utf-8")

    return phase2_output


if __name__ == "__main__":
    result = run_phase2()
    print(json.dumps({
        "sipa_scenarios": len(result["sipa_results"]),
        "ascd_cases": len(result["ascd_results"]),
        "mean_sipa_f1": result["sipa_evaluation"]["mean_f1"],
        "ascd_accuracy": result["ascd_evaluation"]["accuracy"],
    }, indent=2))
