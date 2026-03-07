from logging.config import fileConfig

from sqlalchemy import engine_from_config
from sqlalchemy import pool

from alembic import context

# 🔥 核心修改 1: 导入您的 Base 和 models
# 必须将当前路径加入 sys.path，否则找不到模块
import sys
import os
sys.path.append(os.getcwd())

from database import Base, SQLALCHEMY_DATABASE_URL
import sql_models  # 必须导入 models，否则 Base.metadata 无法识别到表结构

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# 🔥 核心修改 2: 动态设置数据库 URL
# 这样就不需要在 alembic.ini 里硬编码 URL 了，保持单一事实来源
config.set_main_option("sqlalchemy.url", SQLALCHEMY_DATABASE_URL)

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# 🔥 核心修改 3: 设置 target_metadata
# 这是 Alembic 用来对比数据库当前状态和代码中定义状态的关键
target_metadata = Base.metadata

def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        # 对于 SQLite，必须开启 batch 模式才能支持 ALTER TABLE (如删除列)
        render_as_batch=True
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            # 🔥 核心修改 4: SQLite 必须开启 render_as_batch
            render_as_batch=True
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
