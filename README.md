Мои лабораторные работы по интеллектуальному анализу данных в химии. Здесь рассматриваются разные модели машинного обучения для предсказания logBCF.


LogBCF (Log Bioconcentration Factor) — это логарифм биоконцентрационного фактора (BCF), который используется для оценки способности химического вещества накапливаться в живых организмах.

BCF = С(организм) / C (среда)

LogBCF=log_10(BCF)​

BCF — это отношение концентрации вещества в организме к его концентрации в окружающей среде при равновесии.

В хемоинформатике LogBCF часто рассчитывается с помощью моделей структуры-свойства (QSAR), которые связывают молекулярные свойства соединения, такие как гидрофобность (обычно измеряемую как LogP), с его способностью накапливаться в живых организмах.

LogBCF — это важный параметр для оценки экологической безопасности химических веществ. Это показатель, который используется для оценки накопления химического вещества в организмах из окружающей среды, но не напрямую в человеке. Основное его применение связано с экологической токсикологией.

LogBCF связан с липофильностью (через LogP), он иногда используется как дополнительный параметр в рамках QSAR для оценки вероятности биоаккумуляции или токсичности


BCF чаще всего применяется для оценки биоаккумуляции в:
    * Водных организмах: Рыбы, моллюски, ракообразные.
    * Наземных экосистемах: Почвенные микроорганизмы, растения

Вещества с высоким LogBCF имеют тенденцию накапливаться в жировых тканях, что может быть релевантным для человека.
    * Например, такие вещества могут быть медленно метаболизируемыми и плохо выводиться.
    * Лекарства и химические вещества, попадающие в окружающую среду, могут накапливаться в рыбах или других животных, которые являются частью пищевой цепочки человека.