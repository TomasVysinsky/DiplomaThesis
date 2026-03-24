import colorsys
import hashlib
from typing import List, Tuple, Dict, Optional

import torch
from DataModel import LitteringExecution, SensorHeaderMessage, SensorDataMessage, WeightExecution
from DataViewModel import SensorDataMessageWithLittering
import matplotlib.pyplot as plt
import statistics
from collections import defaultdict
from DataViewModel import LoadedDataViewModel, Vehicle, SensorDataMessageWithLittering
#import LiteringCategory
#import LitteringCandidte
#from PredictionToGtMatcher import PreparedLitteringExecution
from RfidHelper import RFIDHelper

#from NonMaximumSupression import LitteringCandidate
import matplotlib.pyplot as plt
#from WindowedDumpDataset import SlidingWindow
from Networks import SolutionApproach
import datetime
import statistics
import numpy as np
import os
import matplotlib.dates as mdates
from vehicle_helper import VehicleHelper
import WindowedLeBasedDataset
#import WindowedLeBasedSmallBigDataset
import bisect
import random

class AttrVisualizationObject:
    def __init__(self):
        self.times_local = []
        self.axis_x_acc_s = []
        self.axis_y_acc_s = []
        self.axis_z_acc_s = []
        self.sig_pwr_s = []
        self.b_s = []
        self.velocity_s = []
        self.rssi_rfid_s = []
        
        self.rfid_present_s = []
        
    def add(self, curr_time: datetime.datetime, tensor: torch.Tensor, y_unit_id : int):
        self.times_local.append(curr_time)
                        
        self.axis_x_acc_s.append((tensor[0])/4 + y_unit_id + 0.5)
        self.axis_y_acc_s.append((tensor[1])/4 + y_unit_id + 0.5)
        self.axis_z_acc_s.append((tensor[2])/4 + y_unit_id + 0.5)
        
        self.sig_pwr_s.append((tensor[3])/4 + y_unit_id + 2)
        
        #self.b_s.append((tensor[4] if tensor.shape[0] > 3 else 0.0 )/4 + y_unit_id + 3)
        
        self.velocity_s.append((tensor[4] if tensor.shape[0] > 5 else 0.0)/4 + y_unit_id + 4)
        
        self.rssi_rfid_s.append((tensor[5] if tensor.shape[0] > 6 else 0.0)/4 + y_unit_id + 5)
        
        self.rfid_present_s.append((tensor[6] if tensor.shape[0] > 11 else 0.0)/4 + y_unit_id + 6)
        

class Visualizer:
    def __init__(self):
        pass
    
    @staticmethod
    def plot_histogram(data, xlabel, ylabel, title, pa_bins, fig_width=10, fig_height=5, log=False, ax=None):
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        ax.hist(data, bins=pa_bins, edgecolor='black', log=log)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True)

        if ax is not None and not plt.fignum_exists(ax.figure.number):
            plt.show()

        return ax
       
    @staticmethod
    def visualize_data_for_vehicle(vehicle : Vehicle, every_minutes : int = 30, save: bool = True):
        for unit_id in vehicle.unit_ids:
            data_by_unit_id = vehicle.data_message_with_literings_by_unit_id[unit_id]
            first_time =  data_by_unit_id[0].data.real_time_computed
            last_time = data_by_unit_id[-1].data.real_time_computed
            
            interval_start = first_time
            interval_end = first_time + datetime.timedelta(minutes = every_minutes)    
            while interval_start < last_time:
                
                Visualizer.visualize_data_with_littering_executions(vehicle, interval_start, interval_end, 1, save)
                 
                interval_start = interval_start + datetime.timedelta(minutes = every_minutes)
                interval_end = interval_end + datetime.timedelta(minutes = every_minutes)
        
    @staticmethod
    def visualize_ground_truth_and_predictions_in_interval(
        vehicle : Vehicle,
        predictions: Dict[str, List[LitteringCandidate]],
        sliding_windows : Dict[str, List[SlidingWindow]],
        stride : int,
        solution_approach : SolutionApproach,
        datetime_start: datetime.datetime,
        datetime_end : datetime.datetime,
        minute_interval: int
    ):  
        current_start = datetime_start
        delta = datetime.timedelta(minutes=minute_interval)

        while current_start < datetime_end:
            current_end = min(current_start + delta, datetime_end)
            

            Visualizer.visualize_ground_truth_and_predictions(vehicle, 
                                                              predictions, 
                                                              sliding_windows, 
                                                              stride, 
                                                              solution_approach, 
                                                              current_start, 
                                                              current_end)
            current_start = current_end
        

    @staticmethod
    def visualize_ground_truth_and_predictions(
        vehicle : Vehicle,
        predictions: Dict[str, List[LitteringCandidate]],
        sliding_windows : Dict[str, List[SlidingWindow]],
        stride : int,
        solution_approach : SolutionApproach,
        datetime_start: datetime.datetime,
        datetime_end : datetime.datetime,
        plot_merged : bool = True
    ):
        visualize_windows_probs = False
        visualize_predictions_as_curve = False 
        
        plt.figure(figsize=(14, 5))
        
        y_levels = []

        for i, unit_id in enumerate(vehicle.unit_ids):
            
            filtered_les = Visualizer.filter_les(vehicle.littering_executions_by_unit_id[unit_id], datetime_start, datetime_end) 
            filtered_candidates : List[LitteringCandidate] = Visualizer.filter_candidates(predictions[unit_id], datetime_start, datetime_end)
            
            
            y_unit_id = 25 - (10 *i ) 
            y_levels.append(y_unit_id)
            
            Visualizer.plot_les(filtered_les, y_unit_id)

           
            if visualize_predictions_as_curve and solution_approach == SolutionApproach.SLIDING_WINDOW_ONE_RESULT:
                #toto plotne len tych ktori presli non maximum supression
                Visualizer.plot_candidates_as_curve(filtered_candidates, y_unit_id)
            else:
                Visualizer.plot_candidates(filtered_candidates, y_unit_id)
                
            if visualize_windows_probs and solution_approach == SolutionApproach.SLIDING_WINDOW_ONE_RESULT:
                #toto plotne aj tych ktori nepresli nms
                Visualizer.plot_sliding_windows_probs(sliding_windows[unit_id], datetime_start, datetime_end, y_unit_id)
                
            filtered_data = Visualizer.filter_data(vehicle.data_message_with_literings_by_unit_id[unit_id], datetime_start, datetime_end)
            
            Visualizer.plot_attributes(filtered_data, y_unit_id, 2)
                
        #vizualizacia zlucenych LE
        if plot_merged and 'all' in vehicle.littering_executions_by_unit_id:
            y_unit_id = y_levels[-1] 
            y_unit_id -= 0.5
            
            filtered_candidates = Visualizer.filter_candidates(predictions['all'], datetime_start, datetime_end)
            Visualizer.plot_candidates(filtered_candidates, y_unit_id)
            
            y_unit_id -= 0.3
            filtered_les = Visualizer.filter_les(vehicle.littering_executions_by_unit_id['all'], datetime_start, datetime_end)
            Visualizer.plot_les(filtered_les, y_unit_id)
            
        # Popisy a legendy
        plt.xlabel("Time")
        plt.yticks(list(y_levels), labels= [ VehicleHelper.get_car_arm(vehicle.ecv, x) for x in vehicle.unit_ids] )
        plt.title(f"Visualization of collection events and predictions {vehicle.ecv}, ({datetime_start} – {datetime_end})")
        plt.ylim(min(y_levels) -1, max(y_levels) + 10)
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show(block= True)
        
        
    @staticmethod
    def visualize_prepared_probs_for_multiclass(
        start: datetime.datetime,
        end: datetime.datetime,
        y_unit_id: float,
        prepared_probs: Dict[datetime.datetime, List[Tuple]],
        treshold: float,
    ) -> None:
        if not prepared_probs:
            return

        y_offset = 6.0
        y_scale = 2.0
        max_gap = datetime.timedelta(seconds=2.0)

        def to_y(val: float) -> float:
            return y_unit_id + y_offset + (val * y_scale)

        # --- threshold line (same Y transform as probs) ------------------------
        if treshold is not None:
            times_in_range = [t for t in prepared_probs.keys() if start <= t <= end]
            if times_in_range:
                t0 = min(times_in_range)
                t1 = max(times_in_range)

                plt.plot(
                    [t0, t1],
                    [to_y(float(treshold)), to_y(float(treshold))],
                    linestyle="--",
                    linewidth=1.0,
                    color="black",
                    alpha=0.7,
                    label=Visualizer.adjust_label("Prepared probs threshold"),
                )

        def iter_class_entries(triples: List[Tuple]) -> List[Tuple[int, int, float]]:
            """Normalize entries to (class_id, window_id, prob)."""
            if not triples:
                return []

            first = triples[0]
            if not isinstance(first, tuple):
                return []

            # Old format: (class_id, window_id, prob)
            if False: # len(first) == 3:
                out: List[Tuple[int, int, float]] = []
                for cls_id, window_id, prob in triples:
                    try:
                        out.append((int(cls_id), int(window_id), float(prob)))
                    except Exception:
                        continue
                return out

            # New format: (window_id, probs_vector) or (window_id, probs_vector, littering_percentage)
            if len(first) >= 2:
                out: List[Tuple[int, int, float]] = []
                for item in triples:
                    if not isinstance(item, tuple) or len(item) < 2:
                        continue
                    window_id, probs_vec = item[0], item[1]
                    try:
                        wid = int(window_id)
                    except Exception:
                        continue

                    if isinstance(probs_vec, torch.Tensor):
                        probs_list = [float(x) for x in probs_vec.detach().cpu().flatten().tolist()]
                    elif isinstance(probs_vec, (list, tuple)):
                        probs_list = [float(x) for x in probs_vec]
                    else:
                        continue

                    for cls_id, prob in enumerate(probs_list):
                        out.append((int(cls_id), wid, float(prob)))
                return out

            return []

        items = [(t, v) for t, v in prepared_probs.items() if start <= t <= end]
        if not items:
            return
        items.sort(key=lambda x: x[0])

        inferred_num_classes: Optional[int] = None
        for _, triples in items:
            normalized = iter_class_entries(triples)
            if not normalized:
                continue
            max_cls_id = max(int(cls_id) for cls_id, _, _ in normalized)
            inferred_num_classes = max((inferred_num_classes or 0), max_cls_id + 1)

        def class_to_color(cls_id: int) -> str:
            if inferred_num_classes == 5:
                # MultiClass single-result (5 classes)
                if cls_id == 0:
                    return "gray"  # Nothing
                if cls_id == 1:
                    return "blue"  # LeftSmallContainer
                if cls_id == 2:
                    return "#42d4f4"  # RightSmallContainer
                if cls_id == 3:
                    return "red"  # BigContainer
                if cls_id == 4:
                    return "green"  # TwoSmallContainers
                return "black"

            # Legacy 3-class mapping
            if cls_id == 1:
                return "blue"  # small
            if cls_id == 2:
                return "red"  # big
            if cls_id == 0:
                return "gray"  # none
            return "black"

        def class_to_linestyle(cls_id: int) -> str:
            if inferred_num_classes == 5 and int(cls_id) in (3, 4):
                return "--"
            return "-"

        # segmentujeme zvlášť pre každú dvojicu (class_id, window_id)
        seg_times_by_key: Dict[Tuple[int, int], List[datetime.datetime]] = defaultdict(list)
        seg_vals_by_key: Dict[Tuple[int, int], List[float]] = defaultdict(list)
        last_t_by_key: Dict[Tuple[int, int], datetime.datetime] = {}

        def class_label(cls_id: int) -> str:
            if inferred_num_classes == 5:
                try:
                    from MultipleCategoriesSingleResultEnum import MultipleCategoriesSingleResultEnum

                    mapping = [
                        MultipleCategoriesSingleResultEnum.Nothing.value,
                        MultipleCategoriesSingleResultEnum.LeftSmallContainer.value,
                        MultipleCategoriesSingleResultEnum.RightSmallContainer.value,
                        MultipleCategoriesSingleResultEnum.BigContainer.value,
                        MultipleCategoriesSingleResultEnum.TwoSmallContainers.value,
                    ]
                    if 0 <= int(cls_id) < len(mapping):
                        return mapping[int(cls_id)]
                except Exception:
                    pass
                return f"cls{cls_id}"

            if cls_id == 0:
                return "none"
            if cls_id == 1:
                return "small"
            if cls_id == 2:
                return "big"
            return f"cls{cls_id}"

        def label_for_key(cls_id: int, window_id: int) -> str:
            return f"Prepared probs {class_label(cls_id)}"

        def flush(key: Tuple[int, int]) -> None:
            ts = seg_times_by_key.get(key, [])
            ys = seg_vals_by_key.get(key, [])
            if len(ts) >= 2:
                cls_id, window_id = key
                plt.plot(
                    ts,
                    ys,
                    linewidth=0.9,
                    color=class_to_color(cls_id),
                    linestyle=class_to_linestyle(cls_id),
                    label=Visualizer.adjust_label(label_for_key(cls_id, window_id)),
                    alpha=0.95,
                )
            seg_times_by_key[key].clear()
            seg_vals_by_key[key].clear()

        for t, triples in items:
            # keď v čase nič nie je, nerob nič; gap sa rieši cez max_gap podľa last_t
            if not triples:
                continue

            normalized = iter_class_entries(triples)
            for cls_id, window_id, prob in normalized:
                key = (int(cls_id), int(window_id))

                prev_t = last_t_by_key.get(key)
                if prev_t is not None and (t - prev_t) > max_gap:
                    flush(key)

                seg_times_by_key[key].append(t)
                seg_vals_by_key[key].append(to_y(float(prob)))
                last_t_by_key[key] = t

        for key in list(seg_times_by_key.keys()):
            flush(key)

       
        
        
    @staticmethod
    def visualize_ground_truth_and_predictions_new(
        vehicle : Vehicle,
        prepared_detections : Dict[str, List[PreparedLitteringExecution]],
        predictions: Dict[str, List[PreparedLitteringExecution]],
        vehicle_littering : LitteringCandidte.VehicleLittering,
        datetime_start: datetime.datetime,
        datetime_end : datetime.datetime,
        treshold : float,
        unit_id_for_vanilla_gradient = None,
        plot_probabilities_from_nn = True,
        additional_info : str = '',
        prepared_probs = None,
        data_messages_override_by_unit_id: Optional[Dict[str, List[SensorDataMessageWithLittering]]] = None,
    ):
        plt.figure(figsize=(14, 5))
        
        y_levels = []
        
        for unit_id in vehicle.unit_ids:
            print(f"Unit id: {unit_id} car arm : {VehicleHelper.get_car_arm(vehicle.ecv, unit_id)} ")

        
        filtered_les_both : Dict[str, List[LitteringExecution]] = {}
        filtered_candidates_both : Dict[str, List[LitteringCandidate]] = {}
        
        for unit_id in vehicle.unit_ids:
            filtered_les_both[unit_id] = []
            filtered_candidates_both[unit_id] = []
        for i, unit_id in enumerate(vehicle.unit_ids):
            other_unit_id = vehicle.unit_ids[0] if unit_id == vehicle.unit_ids[1] else vehicle.unit_ids[1]
            
            filtered_prepared_les : List[PreparedLitteringExecution] = Visualizer.filter_gt_le(prepared_detections[unit_id], datetime_start, datetime_end) 
            
            
           
            for prep in filtered_prepared_les:
                le = LitteringExecution()
                le.timestamp_start = prep.timestamp_start
                le.timestamp_end = prep.timestamp_end
                le.rfid_tag = prep.rfid
                le.is_paired_to_prediction = prep.is_paired
                le.is_delta_rfid = False
                le.trash_can = prep.note + "_" + str(prep.id) + ('_repeated' if prep.is_repeated_lifting else '')
                
                
                filtered_les_both[unit_id].append(le)
                if Visualizer.is_big(prep.category):
                    filtered_les_both[other_unit_id].append(le) 
            
            filtered_prepared_candidates : List[PreparedLitteringExecution] = Visualizer.filter_gt_le(predictions[unit_id], datetime_start, datetime_end)
            for prep in filtered_prepared_candidates:
                cand = LitteringCandidate(prep.timestamp_start, prep.timestamp_end, torch.tensor(1.0))
                cand.is_paired_to_littering_execution = prep.is_paired
                cand.note = prep.category.value + " " + prep.note + '_' + str(prep.id)
                
                filtered_candidates_both[unit_id].append(cand)
                if Visualizer.is_big(prep.category):
                    print(f"Pridala sa cand: start: {prep.timestamp_start}, end: {prep.timestamp_end} do unit id: {other_unit_id}")
                    filtered_candidates_both[other_unit_id].append(cand)
                    
            
        filtered_data_first = None
        for i, unit_id in enumerate(vehicle.unit_ids):
            
            y_unit_id = 25 - (10 *i ) 
            y_levels.append(y_unit_id)
            
            Visualizer.plot_les(filtered_les_both[unit_id], y_unit_id)

            Visualizer.plot_candidates(filtered_candidates_both[unit_id], y_unit_id)
            
                
                
            if data_messages_override_by_unit_id is not None:
                data_source = data_messages_override_by_unit_id.get(unit_id)
                if data_source is None:
                    data_source = vehicle.data_message_with_literings_by_unit_id[unit_id]
            else:
                data_source = vehicle.data_message_with_literings_by_unit_id[unit_id]

            filtered_data = Visualizer.filter_data(data_source, datetime_start, datetime_end)
            if filtered_data_first is None:
                filtered_data_first = filtered_data
            if unit_id_for_vanilla_gradient is None:
                Visualizer.plot_attributes(filtered_data, y_unit_id, 2)
            else:
                Visualizer.plot_vanilla_gradient(i, filtered_data_first, vehicle_littering.vanilla_gradient_output_by_unit_id[unit_id_for_vanilla_gradient], y_unit_id, 2)
            if plot_probabilities_from_nn:
                Visualizer.plot_probabilities_from_nn(filtered_data_first, unit_id, y_unit_id, vehicle_littering, 2, treshold)
                
            if prepared_probs is not None:
                # Backward compatible:
                # - multiclass (two-arm) format: prepared_probs[unit_id] -> Dict[datetime, List[Tuple]]
                # - single-result format: prepared_probs -> Dict[datetime, List[Tuple]] (no unit_id dimension)
                if isinstance(prepared_probs, dict) and unit_id in prepared_probs:
                    probs_for_plot = prepared_probs[unit_id]
                else:
                    probs_for_plot = prepared_probs

                Visualizer.visualize_prepared_probs_for_multiclass(
                    datetime_start,
                    datetime_end,
                    y_unit_id,
                    probs_for_plot,
                    treshold,
                )
                
            Visualizer.plot_information_from_rfid(filtered_data, y_unit_id)
            
        # Popisy a legendy
        plt.xlabel("Time")
        plt.yticks(list(y_levels), labels= [ VehicleHelper.get_car_arm(vehicle.ecv, x) for x in vehicle.unit_ids] )
        plt.title(f"{additional_info}Visualization of collection events and predictions {vehicle.ecv}, ({datetime_start} – {datetime_end})")
        plt.ylim(min(y_levels) -1, max(y_levels) + 10)
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show(block= True)
        
    @staticmethod
    def is_big(category : LiteringCategory.LiteringCategory) -> bool:
        return category.value == LiteringCategory.LiteringCategory.SB.value or category.value == LiteringCategory.LiteringCategory.SA.value
        
    @staticmethod
    def filter_candidates(candidates : List[LitteringCandidate], start_time : datetime.datetime, end_time : datetime.datetime) -> List[LitteringCandidate]:
        filtered_candidates = [
            le for le in candidates
            if (start_time <= le.start_time <= end_time) or (start_time <= le.end_time <= end_time)
        ]
        return filtered_candidates    
    
    @staticmethod
    def filter_les(les : List[LitteringExecution], start_time : datetime.datetime, end_time : datetime.datetime) -> List[LitteringExecution]:
        filtered_les = [
            le for le in les
            if (start_time <= le.timestamp_start <= end_time) or (start_time <= le.timestamp_end <= end_time)
        ]
        return filtered_les
    
    @staticmethod
    def filter_gt_le(les: List[PreparedLitteringExecution], start_time: datetime.datetime, end_time: datetime.datetime) -> List[PreparedLitteringExecution]:
        filtered_les = [
            le for le in les
            if (start_time <= le.timestamp_start <= end_time) or (start_time <= le.timestamp_end <= end_time)
        ]
        return filtered_les
    
    @staticmethod
    def filter_data(
        data: List[SensorDataMessageWithLittering],
        start_time: datetime.datetime,
        end_time: datetime.datetime,
    ) -> List[SensorDataMessageWithLittering]:
        if not data:
            return []

        times = [d.data.real_time_computed for d in data]
        left = bisect.bisect_left(times, start_time)
        right = bisect.bisect_left(times, end_time)  # end is exclusive

        return data[left:right]
    
    @staticmethod
    def plot_candidates_as_curve(filtered_candidates : List[LitteringCandidate], y : float):
        times = [pred.start_time + (pred.end_time - pred.start_time)/2 for pred in filtered_candidates]
        scores = [pred.score.item() for pred in filtered_candidates]

        plt.plot(
            times,
            [y + s / 3 for s in scores],  # výškový offset podľa skóre
            color='red',
            linestyle='-',
            linewidth=1.5,
            marker='o',
            label=Visualizer.adjust_label('CE prediction as curve')
        )
        
    @staticmethod
    def plot_candidates(filtered_candidates: List[LitteringCandidate], y: float):
        base_y = y + 0.1
        for pred in filtered_candidates:
            offset = (pred.score.item() / 3)
            y_plot = base_y + offset

            color = 'red'
            label = 'CE prediction'
            if pred.is_paired_to_littering_execution:
                color = 'purple'
                label = 'paired CE prediction'

            plt.plot(
                [pred.start_time, pred.end_time],
                [y_plot, y_plot],
                color=color,
                linestyle='-',
                linewidth=2,
                label=Visualizer.adjust_label(label)
            )

            # note: malé písmená v strede nad čiarou
            note = pred.note
            if note:
                mid_time = pred.start_time + (pred.end_time - pred.start_time) / 2
                plt.text(
                    mid_time,
                    y_plot + 0.25,
                    str(note),
                    va="bottom",
                    ha="center",
                    color=color,
                    fontsize=6,
                )
        
    
    @staticmethod
    def plot_les(filtered_les : List[LitteringExecution], y : float):
        
        for le in filtered_les:
            y_plot = y if not le.is_delta_rfid else y + 0.2
            #if not le.is_original_prediction:
            #    y_plot += 1.0
            
            is_zero_duration = not ((le.timestamp_end - le.timestamp_start).total_seconds() > 0)
            #color
            label = 'Collection event'
            color='green'
            if 'repeated' in le.trash_can:
                color = '#cfeb34'
            else:
                if le.is_paired_to_prediction:
                    label = 'Paired collection event'
                    color = '#90EE90'
                if is_zero_duration:
                    label = '0 duration collection event'
                    color = 'red'
                elif le.is_delta_rfid:
                    color = 'orange'
                    label = 'Delta RFID collection event'
            
            #end time
            end_time = le.timestamp_end if not is_zero_duration else le.timestamp_end + datetime.timedelta(seconds=1)
            
            #container type
            line_width_le = 2
            is_big_container = not RFIDHelper.is_empty_rfid(le.rfid_tag) and RFIDHelper.is_big_container(le.rfid_tag)
            if is_big_container:
                line_width_le = 5
            #    y_plot += 1.5
            
            plt.plot(
                [le.timestamp_start, end_time],
                [y_plot, y_plot],
                color=color,
                linewidth=line_width_le,
                label=Visualizer.adjust_label(label)
            )
            
            marker_height = 0.1
            plt.plot(
                [le.timestamp_start, le.timestamp_start],
                [y_plot - marker_height, y_plot + marker_height],
                color=color,
                linewidth=line_width_le
            )
            plt.plot(
                [end_time, end_time],
                [y_plot - marker_height, y_plot + marker_height],
                color=color,
                linewidth=line_width_le
            )
            
            if False:
                cas = le.timestamp_start
                final_plot_y = y_plot + 0.5
                duration = (le.timestamp_end - le.timestamp_start).total_seconds()
                plt.text(cas, final_plot_y , f'{le.id}, {le.is_original_prediction}, {le.additional_info}, dur: {duration}',
                        va="center",          # vertikálne zarovnanie ('top', 'center', 'bottom')
                        ha="center",          # horizontálne zarovnanie ('left', 'center', 'right')
                        color=color,          # farba textu
                        fontweight="bold",
                        fontsize=5.5) 
            
            # trash_can: text nad výsypom, centrovaný na stred
            if le.trash_can is not None and str(le.trash_can).strip() != "":
                mid_time = le.timestamp_start + (end_time - le.timestamp_start) / 2
                plt.text(
                    mid_time,
                    y_plot - 0.75,
                    str(le.trash_can),
                    va="bottom",
                    ha="center",
                    color=color,
                    fontweight="bold",
                    fontsize=6,
                )
            
            if not RFIDHelper.is_empty_rfid(le.rfid_tag):
                cas = le.timestamp_start + (le.timestamp_end - le.timestamp_start)/2
                text = RFIDHelper.replace_first_letter_with_x(le.rfid_tag)
                plt.text(cas, y_plot - 1.2 , text,
                        va="center",          # vertikálne zarovnanie ('top', 'center', 'bottom')
                        ha="center",          # horizontálne zarovnanie ('left', 'center', 'right')
                        color=Visualizer.color_from_string_hex_hsv(text),          # farba textu
                        fontweight="bold",
                        fontsize=5.5)
            
    @staticmethod
    def plot_weight_executions(weight_executions: List[WeightExecution], y: float, color: str = 'black'):
        for we in weight_executions:
            is_zero_duration = not ((we.timestamp_end - we.timestamp_start).total_seconds() > 0)
            end_time = we.timestamp_end if not is_zero_duration else we.timestamp_end + datetime.timedelta(seconds=2)
            plt.plot(
                [we.timestamp_start, end_time],
                [y, y],
                color=color,
                linewidth=2,
                label=Visualizer.adjust_label('Weight execution')
            )
            mid_time = we.timestamp_start + (end_time - we.timestamp_start) / 2
            plt.text(
                mid_time,
                y + 0.4,
                f"{we.weight:.0f}",
                ha='center',
                va='bottom',
                fontsize=7,
                fontweight='bold',
                color=color
            )

    @staticmethod
    def plot_sliding_windows_probs(sliding_windows_for_unit_id : List[SlidingWindow], datetime_start : datetime.datetime, datetime_end : datetime.datetime, y_unit_id : int):
        
        filtered_windows : List[SlidingWindow] = [
            p for p in sliding_windows_for_unit_id
            if  datetime_start <= p.start_time() < datetime_end# and p.littering_percentage > 0.0
        ]
        
        for window in filtered_windows:
            base_y = y_unit_id + 0.1
            offset = (window.littering_percentage.item()/3)

            y = base_y + offset

            plt.plot(
                [window.start_time(), window.end_time()],
                [y, y],
                color='yellow',
                linestyle='-',
                linewidth=1,
                label=Visualizer.adjust_label('Sliding window probabilities')
            )
        
    @staticmethod
    def plot_attributes(filtered_data : List[SensorDataMessageWithLittering], y_unit_id : int, dlzka_medzery : float):
        avo = AttrVisualizationObject()
        last_time = None
        for i, data in enumerate(filtered_data):
            curr_time = data.data.real_time_computed
            if last_time is not None and (curr_time - last_time).total_seconds() > dlzka_medzery:
                Visualizer.plot_attrs_from_attr_visualization_object(avo)
                avo = AttrVisualizationObject()
                
            last_time = curr_time
            tensor = data.input_tensor
            
            avo.add(curr_time, tensor, y_unit_id)
        
        Visualizer.plot_attrs_from_attr_visualization_object(avo)
        
    @staticmethod
    def plot_probabilities_from_nn(
        filtered_data: List[SensorDataMessageWithLittering],
        unit_id: str,
        y_unit_id: int,
        vehicle_littering: LitteringCandidte.VehicleLittering,
        gap_seconds: float = 2.0,
        threshold: float = 0.0,
        y_offset: float = 6.0,
        y_scale: float = 2.0,
    ) -> None:
        #print(f"skutocne unit id: {unit_id}")
        #print(f" unit ids z vehicle litering: {[x for x in vehicle_littering.probabilities_by_unit_id.keys()]}")
        
        probs_for_unit_id = vehicle_littering.probabilities_by_unit_id.get(unit_id, {})
        #print(f"unit id: {unit_id}")
        #print(f"Vehicle littering unit ids: {[x for x in vehicle_littering.probabilities_by_unit_id_and_window_id.keys()]}")
        probabilities_by_unit_id_and_window_id = vehicle_littering.probabilities_by_unit_id_and_window_id.get(unit_id, {})
        #for key in probabilities_by_unit_id_and_window_id.keys():
        #    print(f"Probability time key: {key}")
        
        if not probs_for_unit_id or not filtered_data:
            return

        max_gap = datetime.timedelta(seconds=gap_seconds)

        def to_y(val: float) -> float:
            # vyššie + viac "roztiahnuté" na Y osi
            return y_unit_id + y_offset + (val * y_scale)

        # threshold ako vodorovná čiara v rovnakej škále
        if threshold is not None:
            t0 = filtered_data[0].data.real_time_computed
            t1 = filtered_data[-1].data.real_time_computed
            plt.plot(
                [t0, t1],
                [to_y(threshold), to_y(threshold)],
                linestyle="--",
                linewidth=1.0,
                color="black",
                alpha=0.7,
                label=Visualizer.adjust_label("Vehicle littering threshold"),
            )

        seg_times: List[datetime.datetime] = []
        seg_values: List[float] = []

        def flush_segment() -> None:
            if len(seg_times) >= 2:
                plt.plot(
                    seg_times,
                    seg_values,
                    label=Visualizer.adjust_label("Vehicle littering probability"),
                    linewidth=0.9,
                    color="black",
                )
                
        # --- plot probabilities per window_id (colored) -------------------------
        if True and probabilities_by_unit_id_and_window_id:
            print("dostal som sa sem")

            window_ids_by_time: Dict[datetime.datetime, List[str]] = {}
            for w_local in filtered_data:
                t_local = w_local.data.real_time_computed
                per_window = probabilities_by_unit_id_and_window_id.get(t_local)
                if not per_window:
                    continue
                window_ids_by_time[t_local] = list(per_window.keys())

            all_window_ids = set()
            for ids in window_ids_by_time.values():
                all_window_ids.update(ids)

            def rand_color() -> str:
                # stable-enough random color (matplotlib hex)
                return "#{:06x}".format(random.randint(0, 0xFFFFFF))

            color_by_window_id: Dict[str, str] = {wid: rand_color() for wid in all_window_ids}

            def flush_window_segment(window_id: str, ts: List[datetime.datetime], ys: List[float]) -> None:
                if len(ts) >= 2:
                    plt.plot(
                        ts,
                        ys,
                        linewidth=0.9,
                        color=color_by_window_id.get(window_id, "gray"),
                        label=Visualizer.adjust_label(f"Window prob ({window_id})"),
                        alpha=0.95,
                    )

            seg_times_by_wid: Dict[str, List[datetime.datetime]] = defaultdict(list)
            seg_values_by_wid: Dict[str, List[float]] = defaultdict(list)
            last_t_by_wid: Dict[str, datetime.datetime] = {}

            for w_local in filtered_data:
                t_local = w_local.data.real_time_computed
                per_window = probabilities_by_unit_id_and_window_id.get(t_local)

                # time not present -> close all active segments (gap)
                if not per_window:
                    for wid in list(seg_times_by_wid.keys()):
                        flush_window_segment(wid, seg_times_by_wid[wid], seg_values_by_wid[wid])
                        seg_times_by_wid[wid].clear()
                        seg_values_by_wid[wid].clear()
                        last_t_by_wid.pop(wid, None)
                    continue

                # update every window_id that exists at this time
                for wid, v in per_window.items():
                    prev_t = last_t_by_wid.get(wid)
                    if prev_t is not None and (t_local - prev_t) > max_gap:
                        flush_window_segment(wid, seg_times_by_wid[wid], seg_values_by_wid[wid])
                        seg_times_by_wid[wid].clear()
                        seg_values_by_wid[wid].clear()

                    seg_times_by_wid[wid].append(t_local)
                    seg_values_by_wid[wid].append(to_y(float(v)))
                    last_t_by_wid[wid] = t_local

            # flush remaining window segments
            for wid in list(seg_times_by_wid.keys()):
                flush_window_segment(wid, seg_times_by_wid[wid], seg_values_by_wid[wid])
        if True:
            last_t = None
            for w in filtered_data:
                t = w.data.real_time_computed
                

                # ak pre tento čas nie je pravdepodobnosť, ukonči segment (medzera)
                if t not in probs_for_unit_id:
                    flush_segment()
                    seg_times, seg_values = [], []
                    last_t = None
                    continue

                # ak je príliš veľký časový skok, ukonči segment (medzera)
                if last_t is not None and (t - last_t) > max_gap:
                    flush_segment()
                    seg_times, seg_values = [], []

                val = float(probs_for_unit_id[t])
                seg_times.append(t)
                seg_values.append(to_y(val))
                last_t = t

                # posledný segment
            flush_segment()
            
    # --- vanilla gradient -------------------------------------------------
    @staticmethod
    def plot_vanilla_gradient(
        unit_id: int,
        filtered_data : List[SensorDataMessageWithLittering],
        vanilla_gradient_output_by_time: Dict[datetime.datetime, torch.Tensor],
        y_unit_id: int,
        gap_seconds: float = 2.0,
    ) -> None:
        max_gap = datetime.timedelta(seconds=gap_seconds)

        # zoradiť podľa času
        
        #for item in vanilla_gradient_output_by_time.keys():
            #print(f"Key: {item}")
        #print(items)
        avo = AttrVisualizationObject()
        last_time: datetime.datetime = None

        for data_window in filtered_data:
            t = data_window.data.real_time_computed
            tensor = vanilla_gradient_output_by_time[t]
      
            if tensor is None:
                continue
            tensor = tensor.cpu()

            # očakávame (2, 13); vyberieme rameno podľa unit_id
            if tensor.dim() != 2 or tensor.shape[0] <= unit_id:
                print("Dim problem")
                continue

            # uzavri segment keď je veľká medzera
            if last_time is not None and (t - last_time) > max_gap:
                Visualizer.plot_attrs_from_attr_visualization_object(avo)
                avo = AttrVisualizationObject()

            arm_tensor = tensor[unit_id]
            avo.add(t, arm_tensor, y_unit_id)
            last_time = t

        Visualizer.plot_attrs_from_attr_visualization_object(avo)
        
        
        
    @staticmethod
    def color_from_string_hsv(s: str, s_min=0.5, s_max=0.9, v_min=0.6, v_max=0.95):
        hval = hashlib.sha1(s.encode()).hexdigest()  # viac bitov než md5
        n = int(hval, 16)
        # rozdeľ n na kúsky pre H, S, V
        hue = (n & 0xffffffff) / float(0xffffffff)  # 0..1
        sat = s_min + ((n >> 32) % 1000) / 1000.0 * (s_max - s_min)
        val = v_min + ((n >> 42) % 1000) / 1000.0 * (v_max - v_min)
        r,g,b = colorsys.hsv_to_rgb(hue, sat, val)
        return (r,g,b)
           
    @staticmethod 
    def color_from_string_hex_hsv(s: str):
        r,g,b = Visualizer.color_from_string_hsv(s)
        return '#%02x%02x%02x' % (int(r*255), int(g*255), int(b*255))
            
    @staticmethod
    def plot_information_from_rfid(filtered_data : List[SensorDataMessageWithLittering], y_unit_id : float):
        y_tag = y_unit_id + 6
            
        tag_times_big = []
        tag_times_small = []
        
        offset_big = 0.5 
        
        last_rfid = None
        gap_for_plot_rfid_again = 0
        for w in filtered_data:
            if not RFIDHelper.is_empty_rfid(w.data.rfid_tag):
                if RFIDHelper.is_big_container(w.data.rfid_tag):
                    tag_times_big.append(w.data.real_time_computed)
                else:
                    tag_times_small.append(w.data.real_time_computed)
                    
                if last_rfid != w.data.rfid_tag or gap_for_plot_rfid_again > 1:
                    text = RFIDHelper.replace_first_letter_with_x(w.data.rfid_tag)
                    
                    half = len(text) // 2
                    text_split = text[:half] + "\n" + text[half:]
                    
                    plt.text(w.data.real_time_computed, y_tag + 2.1, text,
                        rotation=90,          # otočenie o 90°
                        va="center",          # vertikálne zarovnanie ('top', 'center', 'bottom')
                        ha="center",          # horizontálne zarovnanie ('left', 'center', 'right')
                        color=Visualizer.color_from_string_hex_hsv(text),          # farba textu
                        fontweight="bold",
                        fontsize=5.5)          # veľkosť písma
                    
                    gap_for_plot_rfid_again = 0
                last_rfid = w.data.rfid_tag
            else:
                gap_for_plot_rfid_again += 1

        # Vykresliť veľké nádoby – nčervená
        if tag_times_big:
            plt.scatter(tag_times_big, [y_tag + offset_big]*len(tag_times_big),
                        marker="_", s=50, color="red", label=Visualizer.adjust_label("Big container (in RFID)"))

        # Vykresliť malé nádoby – modrá
        if tag_times_small:
            plt.scatter(tag_times_small, [y_tag]*len(tag_times_small),
                        marker="_", s=50, color="blue", label=Visualizer.adjust_label("Small container (in RFID)"))
            
            
        
        
    @staticmethod
    def save_image(ecv : str, file_name : str):
        cesta = f"{ecv}_visualizations"  # sem zadaj cestu k priečinku

        if not os.path.exists(cesta):
            os.makedirs(cesta)
            print(f"Priečinok '{cesta}' bol vytvorený.")
        
        plt.savefig(f'{cesta}/{file_name}')
        
    @staticmethod
    def visualize_window_new(vehicle : Vehicle, window : WindowedLeBasedSmallBigDataset.SlidingWindow):
        start_time = window.start_time()
        end_time = window.end_time()
        
        Visualizer.visualize_data_with_littering_executions(vehicle, start_time, end_time, 2, False, additional_info=window.container_size.value + '_' + 'flipped' if window.flipped else '_')
        
    @staticmethod
    def visualize_windows_new_data_from_window(vehicle: Vehicle, window : WindowedLeBasedSmallBigDataset.SlidingWindow):
        start_time = window.start_time()
        end_time = window.end_time()
        print("Start time : ", start_time)
        print("End time   : ", end_time)
        
        vehicle_temp = Vehicle(vehicle.ecv, [], [], [])
        vehicle_temp.unit_ids = vehicle.unit_ids
        vehicle_temp.data_message_with_literings_by_unit_id = {}
        vehicle_temp.littering_executions_by_unit_id = vehicle.littering_executions_by_unit_id
        print(f"left count : {len(window.data_messages_left)}")
        #for dm in window.data_messages_left:
        #    print(f" left dm time: {dm.data.real_time_computed}")
        print(f"right count: {len(window.data_messages_right)}")
        #for dm in window.data_messages_right:
        #    print(f" right dm time: {dm.data.real_time_computed}")
        vehicle_temp.data_message_with_literings_by_unit_id[vehicle.unit_ids[0]] = window.data_messages_left 
        
        vehicle_temp.data_message_with_literings_by_unit_id[vehicle.unit_ids[1]] = window.data_messages_right 
        
        Visualizer.visualize_data_with_littering_executions(vehicle_temp, start_time, end_time, 2, False, additional_info=window.container_size.value)
        
    @staticmethod
    def visualize_window(window : WindowedLeBasedDataset.SlidingWindow, agent_logit, plt_rfid_information: bool = True):
        plt.figure(figsize=(14, 6))
        y_levels = []
        any_data = False
        
        unit_ids_labels = ['L', 'R']
        
        data_by_unit_id = []
        data_by_unit_id.append(window.data_messages_left)
        data_by_unit_id.append(window.data_messages_right)
        

        for i, data in enumerate(data_by_unit_id):
            
            y_unit_id = 20 - (10 *i ) 
            y_levels.append(y_unit_id)
                
               
               
                
            filtered_data = data
            if len(filtered_data) == 0:
                continue
            any_data = True

            Visualizer.plot_attributes(filtered_data, y_unit_id, 2)
            #if plt_rfid_information:
                #Visualizer.plot_information_from_rfid(filtered_data, y_unit_id)
        
        filitered_les = [ window.littering_execution ]
        if window.littering_execution.car_arm == 'L':
            Visualizer.plot_les(filitered_les, y_levels[0])
        elif window.littering_execution.car_arm == 'R':
            Visualizer.plot_les(filitered_les, y_levels[1])
        elif window.littering_execution.car_arm == 'B':
            Visualizer.plot_les(filitered_les, y_levels[0])
            Visualizer.plot_les(filitered_les, y_levels[1])
        else:
            raise Exception(f"Car arm bolo nieco ine ako L, R, B: {window.littering_execution.car_arm}")
            
        if any_data:
            # Popisy a legendy
            class_names = ['L', 'R', 'B1', 'B2']
            class_name = class_names[window.result_class] if 0 <= window.result_class < len(class_names) else str(window.result_class)
            
            # Extrahovať najväčší logit a zmapovať na názov
            predicted_class_idx = int(torch.argmax(agent_logit, dim=1).item()) if hasattr(agent_logit, 'argmax') else 0
            predicted_class_name = class_names[predicted_class_idx] if 0 <= predicted_class_idx < len(class_names) else str(predicted_class_idx)
            
            plt.xlabel("Time")
            plt.yticks(list(y_levels), labels = unit_ids_labels)
            plt.title(f"Ground truth: {class_name} | Model predicts: {predicted_class_name}")
            plt.ylim(min(y_levels) -1, max(y_levels) + 10)
            plt.grid(True, linestyle='--', alpha=0.4)
            
            plt.legend(loc='upper right')
            plt.tight_layout()
            plt.show()
        else:
            print(f"Pre okno neexistuju data")
        
    @staticmethod
    def visualize_data_with_littering_executions(
        vehicle : Vehicle,
        datetime_start: datetime.datetime,
        datetime_end : datetime.datetime,
        dlzka_medzery : float,
        save : bool = False,
        plot_les : bool = True,
        plos_rfid_information : bool = True,
        only_first_arm : bool = False,
        additional_info : str = ""
        
    ):
        if additional_info != "":
            additional_info = "_" + additional_info
        if not only_first_arm:
            plt.figure(figsize=(14, 6))
        else:
            plt.figure(figsize=(14, 3))
        
        y_levels = []
        any_data = False
        
        unit_ids = vehicle.unit_ids
        if only_first_arm:
            unit_ids = []
            unit_ids.append(vehicle.unit_ids[0])

        for i, unit_id in enumerate(unit_ids):
            y_unit_id = 20 - (10 *i ) 
            y_levels.append(y_unit_id)


            filitered_les = Visualizer.filter_les(vehicle.littering_executions_by_unit_id[unit_id], datetime_start, datetime_end) 
            if plot_les:
                Visualizer.plot_les(filitered_les, y_unit_id)
            
            filtered_data = Visualizer.filter_data(vehicle.data_message_with_literings_by_unit_id[unit_id], datetime_start, datetime_end)
            if len(filtered_data) == 0 and len(filitered_les) == 0:
                continue
            any_data = True
            
            
            Visualizer.plot_attributes(filtered_data, y_unit_id, dlzka_medzery)
            if plos_rfid_information:
                Visualizer.plot_information_from_rfid(filtered_data, y_unit_id)
            
                
        # Weight executions – zobrazené ku príslušnému unit_id podľa car_arm
        # L → prvý unit_id, R → druhý unit_id, B → obidva
        if y_levels and hasattr(vehicle, 'weight_executions') and vehicle.weight_executions:
            y_we_offset = 4  # výška nad LE líniou daného ramena
            y_first  = y_levels[0] + y_we_offset if len(y_levels) > 0 else None
            y_second = y_levels[1] + y_we_offset if len(y_levels) > 1 else None
            filtered_wes = [
                we for we in vehicle.weight_executions
                if we.timestamp_start <= datetime_end and we.timestamp_end >= datetime_start
            ]
            if filtered_wes:
                any_data = True
                for we in filtered_wes:
                    arm = (we.car_arm or '').upper()
                    we_color = 'red' if arm == 'B' else 'blue'
                    y_targets = []
                    if arm == 'L' and y_first is not None:
                        y_targets.append(y_first)
                    elif arm == 'R' and y_second is not None:
                        y_targets.append(y_second)
                    elif arm == 'B':
                        if y_first is not None:
                            y_targets.append(y_first)
                        if y_second is not None:
                            y_targets.append(y_second)
                    else:
                        raise Exception("Wrong car arm")
                    for y_we in y_targets:
                        Visualizer.plot_weight_executions([we], y_we, we_color)

        if any_data:
            if True:
            # Popisy a legendy
                plt.xlabel("Time")
                plt.yticks(list(y_levels), labels = [ VehicleHelper.get_car_arm(vehicle.ecv, unit_id) for unit_id in unit_ids])
                plt.title(f"Vehicle {VehicleHelper.get_decoded_name(vehicle.ecv)} ({datetime_start} – {datetime_end}){additional_info}")
                plt.ylim(min(y_levels) -1, max(y_levels) + 10)
                plt.grid(True, linestyle='--', alpha=0.4)
                
                plt.legend(loc='upper right')
                plt.tight_layout()
            else:
                plt.xlabel("Time")
                plt.yticks(
                    list(y_levels),
                    labels=[VehicleHelper.get_car_arm(vehicle.ecv, unit_id) for unit_id in vehicle.unit_ids]
                )
                plt.title(f"Visualization {vehicle.ecv} ({datetime_start} – {datetime_end})")
                plt.ylim(min(y_levels) - 1, max(y_levels) + 10)
                plt.grid(True, linestyle='--', alpha=0.4)

                # legenda mimo graf
                plt.legend(
                    loc='upper left',           # pozícia legendy vzhľadom na anchor
                    bbox_to_anchor=(1.05, 1),   # 1.05 = mierne vpravo od grafu, 1 = horný okraj zarovnaný
                    borderaxespad=0.,
                )
                plt.tight_layout()
            
            if save:
                Visualizer.save_image(vehicle.ecv, f"{vehicle.ecv} start: {datetime_start} end: {datetime_end} dlzka medzery: {dlzka_medzery}.png")
            else:
                plt.show(block= True)
        else:
            print(f"Pre cas {datetime_start} - {datetime_end} neexistuju data")
            
    @staticmethod
    def visualize_whole_vehicle_data_line_rfid_and_littering(
        vehicles: List["Vehicle"],
        gap_seconds: int = 2,
        prepared_candidates: Optional[Dict[str, Dict[str, List[PreparedLitteringExecution]]]] = None,
        start: Optional[datetime.datetime] = None,
        end: Optional[datetime.datetime] = None,
    ) -> None:
        max_gap = datetime.timedelta(seconds=gap_seconds)

        def _in_range(t: datetime.datetime) -> bool:
            if start is not None and t < start:
                return False
            if end is not None and t > end:
                return False
            return True

        for vehicle in vehicles:
            # Roztriediť správy podľa dňa a ramena
            by_date: Dict[datetime.date, Dict[str, List["SensorDataMessageWithLittering"]]] = defaultdict(
                lambda: defaultdict(list)
            )

            for unit_id in vehicle.unit_ids:
                wrappers = vehicle.data_message_with_literings_by_unit_id[unit_id]
                for w in wrappers:
                    msg = w.data
                    if msg is None:
                        continue
                    if not _in_range(msg.real_time_computed):
                        continue
                    by_date[msg.real_time_computed.date()][unit_id].append(w)

            # Also include days where only candidates exist (so overlay is visible even if
            # there are no raw data messages in the selected interval).
            if prepared_candidates is not None:
                cand_by_vehicle = prepared_candidates.get(vehicle.ecv, {})
                for unit_id in vehicle.unit_ids:
                    for cand in cand_by_vehicle.get(unit_id, []):
                        # Check overlap with [start, end] if provided
                        if start is not None and cand.timestamp_end < start:
                            continue
                        if end is not None and cand.timestamp_start > end:
                            continue
                        by_date[cand.timestamp_start.date()][unit_id] = by_date[cand.timestamp_start.date()].get(unit_id, [])
                    

            unit_ids = vehicle.unit_ids
            lane_height = 2.4   
            
            for date_key, per_unit in by_date.items():
                plt.figure(figsize=(14, 3 + len(unit_ids)))
                ax = plt.gca()
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
                ax.xaxis_date()
                plt.title(f"{vehicle.ecv} — {date_key.isoformat()}")

                # If a global [start, end] filter is provided, zoom the x-axis to that range
                day_start = datetime.datetime.combine(date_key, datetime.time.min)
                day_end = day_start + datetime.timedelta(days=1)
                x0 = max(day_start, start) if start is not None else day_start
                x1 = min(day_end, end) if end is not None else day_end
                if x1 > x0:
                    plt.xlim(x0, x1)

                y_ticks, y_labels = [], []

                idx = len(unit_ids) - 1
                for unit_id in unit_ids:
                    y_base = idx * lane_height
                    y_data = y_base + 0.2    # dátová čiara
                    y_tag  = y_base + 1.0    # RFID značky
                    y_le   = y_base + 1.8    # littering_execution
                    y_cand = y_base + 2.15   # candidates (above littering_execution)

                    wrappers = per_unit.get(unit_id, [])

                    # --- dátová čiara -----------------------------------------
                    seg_times = []
                    last_t = None
                    for w in wrappers:
                        t = w.data.real_time_computed
                        if last_t is None or (t - last_t) <= max_gap:
                            seg_times.append(t)
                        else:
                            if len(seg_times) >= 2:
                                plt.plot(seg_times, [y_data]*len(seg_times))
                            seg_times = [t]
                        last_t = t
                    if len(seg_times) >= 2:
                        plt.plot(seg_times, [y_data]*len(seg_times))

                    # --- RFID markery podľa veľkosti nádoby ------------------------------------
                    tag_times_big = []
                    tag_times_small = []
                    
                    offset_big = 0.15 

                    for w in wrappers:
                        if not RFIDHelper.is_empty_rfid(w.data.rfid_tag):
                            if RFIDHelper.is_big_container(w.data.rfid_tag):
                                tag_times_big.append(w.data.real_time_computed)
                            else:
                                tag_times_small.append(w.data.real_time_computed)

                    # Vykresliť veľké nádoby
                    if tag_times_big:
                        plt.scatter(tag_times_big, [y_tag + offset_big]*len(tag_times_big),
                                    marker="|", s=220, color="red", label="Big container")

                    # Vykresliť malé nádoby
                    if tag_times_small:
                        plt.scatter(tag_times_small, [y_tag]*len(tag_times_small),
                                    marker="|", s=220, color="blue", label="Small container")
                    # --- littering_execution čiara ----------------------------
                    le_seg = []
                    last_t: Optional[datetime.datetime] = None
                    for w in wrappers:
                        if w.littering_execution_id == -1:
                            # ukončiť segment, ak existuje
                            if len(le_seg) >= 2:
                                plt.plot(le_seg, [y_le]*len(le_seg))
                            le_seg = []
                            last_t = None
                            continue

                        t = w.data.real_time_computed
                        if not le_seg or last_t is None or (t - last_t) <= max_gap:
                            le_seg.append(t)
                        else:
                            plt.plot(le_seg, [y_le]*len(le_seg))
                            le_seg = [t]
                        last_t = t
                    if len(le_seg) >= 2:
                        plt.plot(le_seg, [y_le]*len(le_seg))

                    # --- prepared candidates overlay ---------------------------
                    if prepared_candidates is not None:
                        cand_for_unit = prepared_candidates.get(vehicle.ecv, {}).get(unit_id, [])
                        if cand_for_unit:
                            day_start = datetime.datetime.combine(date_key, datetime.time.min)
                            day_end = day_start + datetime.timedelta(days=1)

                            for cand in cand_for_unit:
                                # plot only candidates overlapping this day
                                if cand.timestamp_end < day_start or cand.timestamp_start >= day_end:
                                    continue

                                # and also overlapping selected [start, end] if provided
                                if start is not None and cand.timestamp_end < start:
                                    continue
                                if end is not None and cand.timestamp_start > end:
                                    continue

                                t0 = max(cand.timestamp_start, day_start)
                                t1 = min(cand.timestamp_end, day_end)
                                if start is not None:
                                    t0 = max(t0, start)
                                if end is not None:
                                    t1 = min(t1, end)

                                # robust: if candidate is a point, draw a marker
                                color = "green" if getattr(cand, "is_paired", False) else "red"
                                label = (
                                    "Candidate paired" if getattr(cand, "is_paired", False)
                                    else "Candidate unpaired"
                                )

                                if t1 <= t0:
                                    plt.plot(
                                        [t0],
                                        [y_cand],
                                        linestyle="None",
                                        marker="o",
                                        markersize=4,
                                        color=color,
                                        label=Visualizer.adjust_label(label),
                                    )
                                else:
                                    plt.plot(
                                        [t0, t1],
                                        [y_cand, y_cand],
                                        color=color,
                                        linewidth=2.0,
                                        alpha=0.95,
                                        label=Visualizer.adjust_label(label),
                                    )

                    # --- oddelovacia čiara medzi ramenami --------------------
                    plt.axhline(y=y_base - 0.3, color="lightgray",
                                linewidth=0.7, zorder=0)

                    # popisky Y-osi
                    y_ticks.extend([y_cand, y_le, y_tag, y_data])
                    y_labels.extend([
                        f"{unit_id} – candidates",
                        f"{unit_id} – littering",
                        f"{unit_id} – tag",
                        f"{unit_id} – data",
                    ])
                    
                    idx-=1

                plt.yticks(y_ticks, y_labels)
                plt.tight_layout()
                plt.show()
            
            
        
    @staticmethod
    def plot_attrs_from_attr_visualization_object(avs : AttrVisualizationObject):
        colors = [
            '#4363d8',  # modrá
            'red',  # žltá
            '#f58231',  # oranžová
            '#911eb4',  # fialová
            '#42d4f4',   # tyrkysová / svetlomodrá,
            '#fabed4'   # ružová
        ]
        
        plt.plot(avs.times_local, avs.axis_x_acc_s, label=Visualizer.adjust_label('Accelerometer x, y, z') , linewidth=0.3, color=colors[0])
        plt.plot(avs.times_local, avs.axis_y_acc_s, label=Visualizer.adjust_label('Accelerometer x, y, z') , linewidth=0.3, color=colors[0])
        plt.plot(avs.times_local, avs.axis_z_acc_s, label=Visualizer.adjust_label('Accelerometer x, y, z') , linewidth=0.3, color = colors[0])
        
        plt.plot(avs.times_local, avs.sig_pwr_s, label=Visualizer.adjust_label('Radar'), linewidth = 0.5, color=colors[1])
        
        #plt.plot(avs.times_local, avs.b_s, label=Visualizer.adjust_label('RFID temperature'), linewidth = 0.5, color=colors[2])
        
        plt.plot(avs.times_local, avs.velocity_s, label=Visualizer.adjust_label('Vehicle velocity'), linewidth = 0.5, color=colors[3])
        plt.plot(avs.times_local, avs.rssi_rfid_s, label=Visualizer.adjust_label('RFID signal strength'), linewidth = 0.5, color=colors[4])
        
        plt.plot(avs.times_local, avs.rfid_present_s, label=Visualizer.adjust_label('RFID hashed'), linewidth = 0.5, color=colors[5])
        
        
    @staticmethod
    def adjust_label(label):
        return label if label not in plt.gca().get_legend_handles_labels()[1] else None
        
        