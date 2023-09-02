"""
Utility for pretty printing config in terminal.
"""
import logging
from typing import Sequence

import rich
import rich.syntax
import rich.tree
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only

logger = logging.getLogger('RichTree')


@rank_zero_only
def print_config_tree(
    cfg: DictConfig,
    print_order: Sequence[str] = (
        'datamodule',
        'train',
        'model'
    ),
    resolve: bool = False
) -> None:
    """
    copied from: https://github.com/ashleve/lightning-hydra-template

    Prints content of DictConfig using Rich library and its tree structure.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.
        print_order (Sequence[str], optional): Determines in what order config components are printed.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """
    style = 'dim'
    tree = rich.tree.Tree('CONFIG', style=style, guide_style=style)

    queue = []

    # add fields from `print_order` to queue
    for field in print_order:
        if field in cfg:
            queue.append(field)
        else:
            logger.warning(
                f'Field "{field}" not found in config. Skipping "{field}" config printing...'
            )

    # add all the other fields to queue (not specified in `print_order`)
    for field in cfg:
        if field not in queue:
            queue.append(field)

    # generate config tree from queue
    for field in queue:
        branch = tree.add(field, style=style, guide_style=style)

        config_group = cfg[field]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)

        branch.add(rich.syntax.Syntax(branch_content, 'yaml'))

    # print config tree
    rich.print(tree)
