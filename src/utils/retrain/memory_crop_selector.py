import pandas as pd
from typing import List, Dict, Tuple, Set

class MemoryCropSelector:
    def __init__(self, mem_scores_df: pd.DataFrame):
        self.mem_scores_df = mem_scores_df.sort_values(by="memory", ascending=False)
        self.mem_scores_df["instance_class"] = self.mem_scores_df["instance_class"].astype(str)
    
    def __call__(self, cls: str, n_crops: int) -> List[str]:
        self.cls_mem_scores_df = self.mem_scores_df[self.mem_scores_df["instance_class"] == cls]
        sorted_crops_to_page = {
            page: crops.sort_values(by="memory", ascending=False)["crop"].tolist()
            for page, crops in self.cls_mem_scores_df.groupby("page")
        }

        n_pages = len(sorted_crops_to_page)
        base, remainder = divmod(n_crops, n_pages)

        selected_crops, extracted_set = self._execute_first_selection_phase(sorted_crops_to_page, base)
        return self._execute_second_selection_phase(selected_crops, extracted_set, remainder)
    
    def _execute_first_selection_phase(self, sorted_crops_to_page: Dict[str, List[str]], base: int) -> Tuple[List[str], Set[str]]:
        selected_crops = []
        for crops in sorted_crops_to_page.values():
            selected_crops.extend(crops[:base])
        return selected_crops, set(selected_crops)

    def _execute_second_selection_phase(self, selected_crops: List[str], extracted_set: Set[str], remainder: int) -> List[str]:
        if remainder == 0: return selected_crops
        
        pages_with_extra = set()
        for _, row in self.cls_mem_scores_df.iterrows():
            if len(pages_with_extra) >= remainder: break
            
            crop, page = row["crop"], row["page"]
            if crop in extracted_set or page in pages_with_extra: continue

            selected_crops.append(crop)
            pages_with_extra.add(page)
        
        return selected_crops